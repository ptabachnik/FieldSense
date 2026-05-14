#!/usr/bin/env python3
"""
Phase 3 robustness/data-efficiency experiment.

This script tests the claim that physics regularization is most useful when
supervised rain labels are sparse, noisy, or partially missing.
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

from .rain_models import RainDataLoss, RainMLP, RainPINNLoss
from .rain_train import (
    calibrate_power_a,
    clone_with_limited_training,
    evaluate_model,
    evaluate_physics_baseline,
    prepare_rain_dataset,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3 sparse/noisy data robustness sweep")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "phase2_validation" / "prepared_with_splits.csv",
        help="Phase 2 prepared table with chronological split column",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "phase3_robustness",
        help="Directory for robustness metrics and plots",
    )
    parser.add_argument(
        "--train-fractions",
        default="1.0,0.5,0.3,0.2,0.1",
        help="Comma-separated fractions of train rows to keep",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated random seeds for subsampling/noise",
    )
    parser.add_argument("--epochs", type=int, default=600, help="Training epochs per model")
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs and fewer fractions")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--power-b", type=float, default=0.8, help="Fixed power-law exponent")
    parser.add_argument(
        "--lambda-physics",
        type=float,
        default=0.01,
        help="PINN physics-loss weight for robustness sweep",
    )
    parser.add_argument(
        "--target-noise-std",
        type=float,
        default=0.0,
        help="Gaussian train-label noise, relative to train target std",
    )
    parser.add_argument(
        "--feature-missing-prob",
        type=float,
        default=0.0,
        help="Probability of masking standardized train features to mean-imputed zero",
    )
    parser.add_argument(
        "--wet-threshold-mm-h",
        type=float,
        default=0.1,
        help="Wet/dry threshold for classification metrics",
    )
    parser.add_argument("--device", default="cpu", help="Torch device")
    return parser.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _train_and_eval_nn(
    model_name: str,
    dataset,
    input_dim: int,
    loss_fn,
    epochs: int,
    lr: float,
    seed: int,
    wet_threshold_mm_h: float,
    device: str,
) -> list[dict[str, float | str]]:
    torch.manual_seed(seed)
    model = RainMLP(input_dim).to(device)
    train_model(model, loss_fn, dataset, n_epochs=epochs, lr=lr)
    metrics, _ = evaluate_model(model_name, model, dataset, wet_threshold_mm_h=wet_threshold_mm_h)
    return metrics


def _plot_rmse(summary: pd.DataFrame, output_path: Path) -> None:
    test = summary[summary["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in test.groupby("model"):
        grouped = group.groupby("train_fraction")["rmse"].mean().reset_index().sort_values("train_fraction")
        ax.plot(grouped["train_fraction"], grouped["rmse"], marker="o", label=model_name)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Training fraction kept")
    ax.set_ylabel("Test RMSE (mm/h)")
    ax.set_title("Data Efficiency Under Missing Training Labels")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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


def _write_report(output_path: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    test = summary[summary["split"] == "test"].copy()
    pivot = (
        test.groupby(["train_fraction", "model"])["rmse"]
        .mean()
        .reset_index()
        .pivot(index="train_fraction", columns="model", values="rmse")
        .reset_index()
        .sort_values("train_fraction", ascending=False)
    )
    if "nn" in pivot.columns and "pinn" in pivot.columns:
        pivot["pinn_rmse_delta_vs_nn"] = pivot["pinn"] - pivot["nn"]
        wins = int((pivot["pinn_rmse_delta_vs_nn"] < 0).sum())
        average_delta = float(pivot["pinn_rmse_delta_vs_nn"].mean())
        best_delta = float(pivot["pinn_rmse_delta_vs_nn"].min())
    else:
        wins = 0
        average_delta = float("nan")
        best_delta = float("nan")

    if wins > 0:
        conclusion = (
            f"The PINN beats the pure NN in {wins}/{len(pivot)} sparse-data settings "
            f"(best RMSE delta {best_delta:.6g})."
        )
    else:
        conclusion = (
            "The PINN does not beat the pure NN in this robustness setting. "
            f"Average PINN-minus-NN RMSE delta is {average_delta:.6g}, so the physics "
            "loss is close but not yet beneficial."
        )

    lines = [
        "# Phase 3 Robustness/Data-Efficiency Report",
        "",
        f"Input: `{args.input}`",
        f"Train fractions: `{args.train_fractions}`",
        f"Seeds: `{args.seeds}`",
        f"Target noise std: `{args.target_noise_std}`",
        f"Feature missing probability: `{args.feature_missing_prob}`",
        f"PINN lambda: `{args.lambda_physics}`",
        "",
        "## Test RMSE by Training Fraction",
        "",
        _dataframe_to_markdown(pivot),
        "",
        "## Result Summary",
        "",
        f"- {conclusion}",
        "",
        "## Interpretation",
        "",
        "- Rows are randomly removed from the training split only; validation and test stay clean.",
        "- Target noise and feature missingness, if enabled, are applied only to training data.",
        "- Negative `pinn_rmse_delta_vs_nn` means the PINN beat the pure NN for that sparse-data setting.",
        "- This experiment directly tests the sparse/noisy-data claim from Phase 1 on real CML/gauge data.",
        "",
        "## Visual Artifacts",
        "",
        "- `test_rmse_vs_train_fraction.png`: test RMSE versus remaining train fraction.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = _parse_float_list(args.train_fractions)
    seeds = _parse_int_list(args.seeds)
    if args.quick:
        fractions = [1.0, 0.3, 0.1]
        args.epochs = min(args.epochs, 300)

    base_dataset = prepare_rain_dataset(args.input, device=args.device)
    input_dim = len(base_dataset.feature_columns)
    rows: list[dict[str, float | str]] = []

    for seed in seeds:
        for fraction in fractions:
            scenario_dataset = clone_with_limited_training(
                base_dataset,
                train_fraction=fraction,
                seed=seed,
                target_noise_std=args.target_noise_std,
                feature_missing_prob=args.feature_missing_prob,
            )
            power_a = calibrate_power_a(scenario_dataset, args.power_b)

            physics_metrics, _ = evaluate_physics_baseline(
                scenario_dataset,
                power_a=power_a,
                power_b=args.power_b,
                wet_threshold_mm_h=args.wet_threshold_mm_h,
            )
            for row in physics_metrics:
                row.update({"train_fraction": fraction, "seed": seed, "power_a": power_a})
                rows.append(row)

            nn_metrics = _train_and_eval_nn(
                "nn",
                scenario_dataset,
                input_dim,
                RainDataLoss(),
                args.epochs,
                args.lr,
                seed,
                args.wet_threshold_mm_h,
                args.device,
            )
            for row in nn_metrics:
                row.update({"train_fraction": fraction, "seed": seed, "power_a": power_a})
                rows.append(row)

            pinn_metrics = _train_and_eval_nn(
                "pinn",
                scenario_dataset, 
                input_dim,
                RainPINNLoss(power_a, args.power_b, lambda_physics=args.lambda_physics),
                args.epochs,
                args.lr,
                seed,
                args.wet_threshold_mm_h,
                args.device,
            )
            for row in pinn_metrics:
                row.update({"train_fraction": fraction, "seed": seed, "power_a": power_a})
                rows.append(row)

    summary = pd.DataFrame(rows)
    summary["target_noise_std"] = args.target_noise_std
    summary["feature_missing_prob"] = args.feature_missing_prob
    summary.to_csv(output_dir / "robustness_metrics.csv", index=False)
    _plot_rmse(summary, output_dir / "test_rmse_vs_train_fraction.png")
    _write_report(output_dir / "robustness_report.md", summary, args)

    test = summary[summary["split"] == "test"]
    print("\nRobustness test RMSE")
    print("-" * 72)
    print(
        test.pivot_table(index="train_fraction", columns="model", values="rmse", aggfunc="mean")
        .sort_index(ascending=False)
        .to_string(float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nSaved robustness report: {output_dir / 'robustness_report.md'}")


if __name__ == "__main__":
    main()
