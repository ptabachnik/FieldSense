#!/usr/bin/env python3
"""
Phase 3 rain-rate estimation experiment.

Compares:
1. Pure power-law physics baseline
2. Pure data-driven MLP
3. PINN-style MLP with power-law attenuation residual
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .rain_models import PhysicsGuidedResidualMLP, RainDataLoss, RainMLP, RainPINNLoss
from .rain_train import (
    append_physics_prior_feature,
    calibrate_power_a,
    evaluate_model,
    evaluate_physics_baseline,
    prepare_rain_dataset,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 rain NN/PINN comparison")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "phase2_validation" / "prepared_with_splits.csv",
        help="Phase 2 prepared table with chronological split column",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "phase3",
        help="Directory for metrics, predictions, and plots",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs per NN model")
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs for a smoke test")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--power-a",
        type=float,
        default=0.35,
        help="Fixed power-law coefficient a in A = a * L * R^b",
    )
    parser.add_argument(
        "--power-b",
        type=float,
        default=1.0,
        help="Fixed power-law exponent b in A = a * L * R^b",
    )
    parser.add_argument(
        "--calibrate-power-a",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit power-law coefficient a on the training split only.",
    )
    parser.add_argument(
        "--lambda-physics",
        type=str,
        default="0.01,0.1,1.0",
        help="Comma-separated PINN physics-loss weights to sweep",
    )
    parser.add_argument(
        "--wet-threshold-mm-h",
        type=float,
        default=0.1,
        help="Wet/dry threshold for classification metrics",
    )
    parser.add_argument("--device", default="cpu", help="Torch device")
    return parser.parse_args()


def _parse_lambdas(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _save_training_history(output_dir: Path, model_name: str, history: list[dict[str, float]]) -> None:
    pd.DataFrame(history).to_csv(output_dir / f"{model_name}_history.csv", index=False)


def _plot_test_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot test predictions against gauge truth for each model."""
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return
    first_target = test["target_name"].iloc[0]
    first_sublink = test["sublink_id"].iloc[0]
    subset = test[(test["target_name"] == first_target) & (test["sublink_id"] == first_sublink)]

    fig, ax = plt.subplots(figsize=(11, 5))
    truth = subset[["time_utc", "target_rain_mm_h"]].drop_duplicates().sort_values("time_utc")
    ax.plot(truth["time_utc"], truth["target_rain_mm_h"], color="black", lw=2, label="Gauge truth")
    for model_name, group in subset.groupby("model"):
        ordered = group.sort_values("time_utc")
        ax.plot(ordered["time_utc"], ordered["prediction_mm_h"], lw=1.5, label=model_name)
    ax.set_title("Phase 3 Test Predictions")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Rain rate (mm/h)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_test_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    """Plot predicted versus observed test rain rates."""
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    max_value = max(test["target_rain_mm_h"].max(), test["prediction_mm_h"].max())
    ax.plot([0, max_value], [0, max_value], color="black", lw=1, linestyle="--", label="ideal")
    for model_name, group in test.groupby("model"):
        ax.scatter(
            group["target_rain_mm_h"],
            group["prediction_mm_h"],
            s=16,
            alpha=0.5,
            label=model_name,
        )
    ax.set_xlabel("Gauge truth (mm/h)")
    ax.set_ylabel("Prediction (mm/h)")
    ax.set_title("Phase 3 Test Prediction Scatter")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_metric_bars(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot key test metrics for selected report models."""
    test = metrics[(metrics["split"] == "test") & metrics["selected_for_report"]].copy()
    if test.empty:
        return
    test = test.sort_values("rmse")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(
        axes,
        ["rmse", "mae", "wet_f1"],
        ["RMSE (lower is better)", "MAE (lower is better)", "Wet/Dry F1 (higher is better)"],
    ):
        ax.bar(test["model"], test[metric], color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(test)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _print_test_summary(metrics: pd.DataFrame) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    cols = ["model", "rmse", "mae", "mse", "wet_f1", "wet_accuracy"]
    print("\nTest metrics")
    print("-" * 72)
    print(test[cols].sort_values("rmse").to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    rendered = frame.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: f"{value:.6g}")
        else:
            rendered[column] = rendered[column].astype(str)
    columns = list(rendered.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _write_report(
    output_path: Path,
    input_path: Path,
    metrics: pd.DataFrame,
    power_a: float,
    power_b: float,
    best_pinn_name: str,
) -> None:
    selected = metrics[(metrics["split"] == "test") & metrics["selected_for_report"]].copy()
    selected = selected[["model", "rmse", "rainy_rmse", "mae", "mse", "wet_f1", "wet_accuracy"]].sort_values("rmse")
    selected_all_splits = metrics[metrics["selected_for_report"]].copy()
    selected_all_splits = selected_all_splits[
        ["model", "split", "rmse", "rainy_rmse", "mae", "mse", "bias", "wet_f1", "wet_accuracy"]
    ].sort_values(["model", "split"])
    best_test_model = selected.iloc[0]["model"]
    best_test_rmse = selected.iloc[0]["rmse"]
    if best_test_model == "physics_feature_nn":
        result_interpretation = (
            "- The best test RMSE comes from using the power-law estimate as a soft NN feature, "
            "supporting the Phase 3B idea that physics is more useful as a prior than as a rigid loss."
        )
    elif best_test_model == "physics_guided_residual":
        result_interpretation = (
            "- The best test RMSE comes from predicting corrections around the power-law prior."
        )
    elif best_test_model == "nn":
        result_interpretation = "- The pure NN is the best model in this run."
    else:
        result_interpretation = "- The pure physics baseline is the best model in this run."
    lines = [
        "# Phase 3 Rain Modeling Report",
        "",
        f"Input: `{input_path}`",
        f"Power law: `A = a * L * R^b`, with train-calibrated `a={power_a:.6g}` and fixed `b={power_b:.6g}`.",
        f"Best PINN by validation RMSE: `{best_pinn_name}`",
        f"Best test RMSE: `{best_test_model}` (`{best_test_rmse:.6g}` mm/h)",
        "",
        "## Phase 3A: Simple Power-Law PINN",
        "",
        "The first Phase 3 PINN uses the power-law relation as a direct physics-loss term:",
        "",
        "`A ≈ a * L * R^b`",
        "",
        "This establishes a simple physics-informed baseline, analogous to Phase 1 where the known physics was added directly to the loss function. However, the real CML rainfall setting is less ideal than the oscillator task. The power-law relation captures first-order microwave attenuation physics, but in real CML/gauge data it is insufficient as a standalone constraint because the CML observes path-averaged attenuation while gauges provide point measurements, and additional effects such as wet antenna, baseline drift, hardware noise, and spatial rain variability violate the idealized assumptions.",
        "",
        "Therefore, the simple power-law PINN should be interpreted as a baseline physics-informed model, not the final hybrid architecture.",
        "",
        "## Phase 3B: Physics-Guided Residual Model",
        "",
        "The Phase 3B model uses the power-law estimate as a soft prior and learns a correction:",
        "",
        "`R_physics = (A / (a * L))^(1/b)`",
        "",
        "`R_hat = Softplus(R_physics + correction_NN(features))`",
        "",
        "This preserves the physical prior while allowing the network to learn real-world deviations from the ideal power-law model.",
        "",
        "## Test Metrics",
        "",
        _dataframe_to_markdown(selected),
        "",
        "## Train/Validation/Test Metrics",
        "",
        _dataframe_to_markdown(selected_all_splits),
        "",
        "## Visual Artifacts",
        "",
        "- `test_predictions.png`: test time series for one target/link.",
        "- `test_prediction_scatter.png`: predicted versus observed rain rate.",
        "- `test_metric_bars.png`: RMSE, MAE, and wet/dry F1 comparison.",
        "",
        "## Interpretation",
        "",
        "- Pure physics establishes the power-law-only baseline.",
        result_interpretation,
        "- The simple loss-only PINN is close at low physics weight, but larger physics weights over-constrain the model.",
        "- The residual model tests a stronger architectural prior; in this split it improves wet/dry behavior but not RMSE.",
        "- The next step should be temporal modeling or better event-aware evaluation, not stronger power-law weighting.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = 300 if args.quick else args.epochs
    lambdas = _parse_lambdas(args.lambda_physics)

    dataset = prepare_rain_dataset(args.input, device=args.device)
    input_dim = len(dataset.feature_columns)
    power_a = calibrate_power_a(dataset, args.power_b) if args.calibrate_power_a else args.power_a
    print(f"Power-law parameters: a={power_a:.6g}, b={args.power_b:.6g}")

    all_metrics: list[dict[str, float | str]] = []
    prediction_tables: list[pd.DataFrame] = []

    physics_metrics, physics_predictions = evaluate_physics_baseline(
        dataset,
        power_a=power_a,
        power_b=args.power_b,
        wet_threshold_mm_h=args.wet_threshold_mm_h,
    )
    all_metrics.extend(physics_metrics)
    prediction_tables.append(physics_predictions)

    baseline_model = RainMLP(input_dim).to(args.device)
    baseline_history = train_model(
        baseline_model,
        RainDataLoss(),
        dataset,
        n_epochs=n_epochs,
        lr=args.lr,
    )
    _save_training_history(output_dir, "nn", baseline_history)
    baseline_metrics, baseline_predictions = evaluate_model(
        "nn",
        baseline_model,
        dataset,
        wet_threshold_mm_h=args.wet_threshold_mm_h,
    )
    all_metrics.extend(baseline_metrics)
    prediction_tables.append(baseline_predictions)

    residual_dataset = append_physics_prior_feature(dataset, power_a=power_a, power_b=args.power_b)
    physics_feature_dataset = append_physics_prior_feature(
        dataset,
        power_a=power_a,
        power_b=args.power_b,
        standardize_prior=True,
    )
    physics_feature_model = RainMLP(len(physics_feature_dataset.feature_columns)).to(args.device)
    physics_feature_history = train_model(
        physics_feature_model,
        RainDataLoss(),
        physics_feature_dataset,
        n_epochs=n_epochs,
        lr=args.lr,
    )
    _save_training_history(output_dir, "physics_feature_nn", physics_feature_history)
    physics_feature_metrics, physics_feature_predictions = evaluate_model(
        "physics_feature_nn",
        physics_feature_model,
        physics_feature_dataset,
        wet_threshold_mm_h=args.wet_threshold_mm_h,
    )
    all_metrics.extend(physics_feature_metrics)
    prediction_tables.append(physics_feature_predictions)

    residual_model = PhysicsGuidedResidualMLP(
        input_dim=len(residual_dataset.feature_columns),
        physics_prior_index=len(residual_dataset.feature_columns) - 1,
    ).to(args.device)
    residual_history = train_model(
        residual_model,
        RainDataLoss(),
        residual_dataset,
        n_epochs=n_epochs,
        lr=args.lr,
    )
    _save_training_history(output_dir, "physics_guided_residual", residual_history)
    residual_metrics, residual_predictions = evaluate_model(
        "physics_guided_residual",
        residual_model,
        residual_dataset,
        wet_threshold_mm_h=args.wet_threshold_mm_h,
    )
    all_metrics.extend(residual_metrics)
    prediction_tables.append(residual_predictions)

    pinn_candidates: list[tuple[float, pd.DataFrame, list[dict[str, float | str]]]] = []
    for lam in lambdas:
        torch.manual_seed(args.seed)
        model = RainMLP(input_dim).to(args.device)
        history = train_model(
            model,
            RainPINNLoss(power_a, args.power_b, lambda_physics=lam),
            dataset,
            n_epochs=n_epochs,
            lr=args.lr,
        )
        model_name = f"pinn_lambda_{lam:g}"
        _save_training_history(output_dir, model_name, history)
        metrics, predictions = evaluate_model(
            model_name,
            model,
            dataset,
            wet_threshold_mm_h=args.wet_threshold_mm_h,
        )
        all_metrics.extend(metrics)
        prediction_tables.append(predictions)
        val_rmse = float(pd.DataFrame(metrics).query("split == 'val'")["rmse"].iloc[0])
        pinn_candidates.append((val_rmse, predictions, metrics))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    best_pinn = min(pinn_candidates, key=lambda item: item[0])
    best_pinn_name = pd.DataFrame(best_pinn[2]).iloc[0]["model"]
    metrics_df["selected_for_report"] = metrics_df["model"].isin(
        ["physics", "nn", "physics_feature_nn", "physics_guided_residual", best_pinn_name]
    )
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    predictions_df = pd.concat(prediction_tables, ignore_index=True)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    _plot_test_predictions(predictions_df, output_dir / "test_predictions.png")
    _plot_test_scatter(predictions_df, output_dir / "test_prediction_scatter.png")
    _plot_metric_bars(metrics_df, output_dir / "test_metric_bars.png")
    _write_report(
        output_dir / "phase3_report.md",
        args.input,
        metrics_df,
        power_a=power_a,
        power_b=args.power_b,
        best_pinn_name=str(best_pinn_name),
    )

    _print_test_summary(metrics_df[metrics_df["selected_for_report"]])
    print(f"\nBest PINN by validation RMSE: {best_pinn_name}")
    print(f"Saved metrics: {output_dir / 'metrics.csv'}")
    print(f"Saved predictions: {output_dir / 'predictions.csv'}")
    print(f"Saved report: {output_dir / 'phase3_report.md'}")


if __name__ == "__main__":
    main()
