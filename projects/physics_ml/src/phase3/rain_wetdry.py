#!/usr/bin/env python3
"""
Phase 3E wet/dry classification experiment.

Uses the same gauge-level spatial table as Phase 3C, but trains a dedicated
classifier instead of thresholding a rain-rate regressor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .rain_models import WetDryMLP
from .rain_spatial import build_spatial_wide_table, prepare_spatial_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3E wet/dry classification")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "phase2_spatial_validation" / "prepared_with_splits.csv",
        help="Validated long table with multiple CML links per target",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "phase3e_wetdry")
    parser.add_argument("--max-links-per-target", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--power-b", type=float, default=0.8)
    parser.add_argument("--wet-threshold-mm-h", type=float, default=0.1)
    parser.add_argument(
        "--threshold-strategy",
        choices=["fixed", "train", "val"],
        default="fixed",
        help="How to choose probability threshold. Fixed uses 0.5.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap resamples for confidence intervals on F1/precision/recall.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute wet/dry classification metrics."""
    y_true = labels.astype(bool).reshape(-1)
    y_pred = (probabilities.reshape(-1) >= threshold)
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    tn = float(np.logical_and(~y_true, ~y_pred).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "n_samples": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "predicted_positive_count": int(y_pred.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def bootstrap_binary_metric_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap confidence intervals for sparse wet/dry metrics."""
    labels = labels.astype(float).reshape(-1)
    probabilities = probabilities.reshape(-1)
    if samples <= 0 or len(labels) == 0:
        return {
            "f1_ci_low": float("nan"),
            "f1_ci_high": float("nan"),
            "precision_ci_low": float("nan"),
            "precision_ci_high": float("nan"),
            "recall_ci_low": float("nan"),
            "recall_ci_high": float("nan"),
        }

    rng = np.random.default_rng(seed)
    values = {"f1": [], "precision": [], "recall": []}
    for _ in range(samples):
        idx = rng.integers(0, len(labels), size=len(labels))
        metrics = binary_metrics(labels[idx], probabilities[idx], threshold)
        for key in values:
            values[key].append(metrics[key])

    return {
        "f1_ci_low": float(np.percentile(values["f1"], 2.5)),
        "f1_ci_high": float(np.percentile(values["f1"], 97.5)),
        "precision_ci_low": float(np.percentile(values["precision"], 2.5)),
        "precision_ci_high": float(np.percentile(values["precision"], 97.5)),
        "recall_ci_low": float(np.percentile(values["recall"], 2.5)),
        "recall_ci_high": float(np.percentile(values["recall"], 97.5)),
    }


def tune_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select threshold maximizing validation F1."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        f1 = binary_metrics(labels, probabilities, threshold)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def make_binary_target(dataset, wet_threshold_mm_h: float, split: str) -> torch.Tensor:
    values = getattr(dataset, split).target
    return (values > wet_threshold_mm_h).float()


def train_classifier(dataset, epochs: int, lr: float, seed: int, wet_threshold_mm_h: float, device: str) -> WetDryMLP:
    """Train classifier with positive-class weighting and validation early stopping."""
    torch.manual_seed(seed)
    model = WetDryMLP(dataset.train.features.shape[1], hidden_dims=[64, 64]).to(device)
    y_train = make_binary_target(dataset, wet_threshold_mm_h, "train")
    y_val = make_binary_target(dataset, wet_threshold_mm_h, "val")
    positives = y_train.sum()
    negatives = y_train.numel() - positives
    pos_weight = torch.clamp(negatives / (positives + 1e-8), min=1.0, max=50.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=lr)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    patience = 100
 
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(dataset.train.features)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(dataset.val.features), y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model


def evaluate_classifier(
    model_name: str,
    model: WetDryMLP,
    dataset,
    wet_threshold_mm_h: float,
    threshold_strategy: str = "fixed",
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> tuple[list[dict[str, float | str]], pd.DataFrame]:
    """Evaluate classifier on train/val/test."""
    model.eval()
    outputs = {}
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            split_data = getattr(dataset, split)
            prob = torch.sigmoid(model(split_data.features)).cpu().numpy().reshape(-1)
            labels = (split_data.target.cpu().numpy().reshape(-1) > wet_threshold_mm_h).astype(float)
            outputs[split] = (labels, prob, split_data.frame.copy())

    if threshold_strategy == "fixed":
        threshold = 0.5
    elif threshold_strategy == "train":
        threshold = tune_threshold(outputs["train"][0], outputs["train"][1])
    elif threshold_strategy == "val":
        threshold = tune_threshold(outputs["val"][0], outputs["val"][1])
    else:
        raise ValueError(f"Unknown threshold strategy: {threshold_strategy}")
    rows = []
    predictions = []
    for split, (labels, prob, frame) in outputs.items():
        row: dict[str, float | str] = {"model": model_name, "split": split, "threshold": threshold}
        row.update(binary_metrics(labels, prob, threshold))
        row.update(
            bootstrap_binary_metric_ci(
                labels,
                prob,
                threshold,
                samples=bootstrap_samples,
                seed=seed + sum(ord(ch) for ch in f"{model_name}:{split}"),
            )
        )
        rows.append(row)
        frame["model"] = model_name
        frame["wet_label"] = labels
        frame["wet_probability"] = prob
        frame["wet_prediction"] = (prob >= threshold).astype(float)
        predictions.append(frame)
    return rows, pd.concat(predictions, ignore_index=True)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    rendered = frame.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: f"{value:.6g}")
        else:
            rendered[column] = rendered[column].astype(str)
    columns = list(rendered.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def plot_wetdry_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    test = metrics[metrics["split"] == "test"].sort_values("f1", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(
        axes,
        ["f1", "precision", "recall"],
        ["Wet/Dry F1", "Precision", "Recall"],
    ):
        ax.bar(test["model"], test[metric])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(output_path: Path, metrics: pd.DataFrame, input_path: Path, max_links: int) -> None:
    test = metrics[metrics["split"] == "test"].sort_values("f1", ascending=False)
    single_f1 = float(test[test["model"] == "single_link_classifier"]["f1"].iloc[0])
    best = test.iloc[0]
    improvement = (float(best["f1"]) - single_f1) / (single_f1 + 1e-8) * 100.0
    lines = [
        "# Phase 3E Wet/Dry Classification Report",
        "",
        f"Input: `{input_path}`",
        f"Links per target: `{max_links}`",
        f"Threshold strategy: `{metrics['threshold_strategy'].iloc[0]}`",
        "",
        "## Test Metrics",
        "",
        _dataframe_to_markdown(
            test[
                [
                    "model",
                    "f1",
                    "f1_ci_low",
                    "f1_ci_high",
                    "precision",
                    "precision_ci_low",
                    "precision_ci_high",
                    "recall",
                    "recall_ci_low",
                    "recall_ci_high",
                    "accuracy",
                    "positive_count",
                    "predicted_positive_count",
                    "threshold",
                    "tp",
                    "fp",
                    "fn",
                ]
            ]
        ),
        "",
        "## Result Summary",
        "",
        f"- Best model: `{best['model']}` with F1 `{float(best['f1']):.6g}`.",
        f"- 95% bootstrap F1 CI: `[{float(best['f1_ci_low']):.3g}, {float(best['f1_ci_high']):.3g}]`.",
        f"- Test positives: `{int(best['positive_count'])}` wet samples.",
        f"- Improvement vs one-link classifier: `{improvement:.3g}%`.",
        "- This directly evaluates wet/dry classification instead of relying only on thresholded rain-rate regression outputs.",
        "",
        "## Visual Artifacts",
        "",
        "- `wetdry_metric_bars.png`: F1, precision, and recall comparison.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = min(args.epochs, 300) if args.quick else args.epochs
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wide, _ = build_spatial_wide_table(args.input, args.power_b, args.max_links_per_target)
    wide.to_csv(output_dir / "wetdry_wide_table.csv", index=False)

    modes = {
        "single_link_classifier": "single_link",
        "single_link_physics_classifier": "single_link_physics",
        "spatial_classifier": "spatial",
        "spatial_physics_classifier": "spatial_physics",
    }
    rows = []
    predictions = []
    for model_name, mode in modes.items():
        dataset = prepare_spatial_dataset(wide, mode=mode, max_links_per_target=args.max_links_per_target, device=args.device)
        model = train_classifier(dataset, epochs=epochs, lr=args.lr, seed=args.seed, wet_threshold_mm_h=args.wet_threshold_mm_h, device=args.device)
        metrics, pred = evaluate_classifier(
            model_name,
            model,
            dataset,
            args.wet_threshold_mm_h,
            threshold_strategy=args.threshold_strategy,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for row in metrics:
            row["threshold_strategy"] = args.threshold_strategy
        rows.extend(metrics)
        predictions.append(pred)

    metrics_df = pd.DataFrame(rows)
    predictions_df = pd.concat(predictions, ignore_index=True)
    metrics_df.to_csv(output_dir / "wetdry_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "wetdry_predictions.csv", index=False)
    plot_wetdry_metrics(metrics_df, output_dir / "wetdry_metric_bars.png")
    write_report(output_dir / "wetdry_report.md", metrics_df, args.input, args.max_links_per_target)

    test = metrics_df[metrics_df["split"] == "test"].sort_values("f1", ascending=False)
    print("\nWet/dry test metrics")
    print("-" * 72)
    print(
        test[
            [
                "model",
                "f1",
                "f1_ci_low",
                "f1_ci_high",
                "precision",
                "recall",
                "positive_count",
                "accuracy",
                "threshold",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nSaved wet/dry report: {output_dir / 'wetdry_report.md'}")


if __name__ == "__main__":
    main()
