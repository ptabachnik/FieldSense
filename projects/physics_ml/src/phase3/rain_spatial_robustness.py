#!/usr/bin/env python3
"""
Spatial robustness/data-efficiency experiment.

Tests whether spatial+physics models degrade more gracefully than single-link
models when training data is reduced. This directly addresses the Metric 2
data-efficiency claim using the spatial configuration that succeeded in Phase 3D.
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

from .rain_spatial import (
    SpatialDataset,
    SpatialSplit,
    build_spatial_wide_table,
    evaluate_spatial_model,
    prepare_spatial_dataset,
    train_spatial_model,
)
from .rain_train import rain_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial robustness under sparse training data")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "phase2_spatial_validation" / "prepared_with_splits.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "phase3_spatial_robustness")
    parser.add_argument("--max-links-per-target", type=int, default=2)
    parser.add_argument(
        "--train-fractions",
        default="1.0,0.5,0.3,0.2,0.1",
        help="Comma-separated fractions of training rows to keep",
    )
    parser.add_argument("--seeds", default="42,7,123", help="Comma-separated random seeds")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--power-b", type=float, default=0.8)
    parser.add_argument("--rain-weight-alpha", type=float, default=2.0)
    parser.add_argument("--wet-threshold-mm-h", type=float, default=0.1)
    parser.add_argument("--target-noise-std", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    return [float(p.strip()) for p in raw.split(",") if p.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def limit_spatial_training(
    dataset: SpatialDataset,
    fraction: float,
    seed: int,
    target_noise_std: float = 0.0,
) -> SpatialDataset:
    """Drop training rows and optionally add label noise; val/test stay clean."""
    gen = torch.Generator(device=dataset.train.features.device)
    gen.manual_seed(seed)

    n = dataset.train.features.shape[0]
    n_keep = max(1, int(n * fraction))
    indices = torch.randperm(n, generator=gen, device=dataset.train.features.device)[:n_keep]
    indices, _ = torch.sort(indices)

    features = dataset.train.features[indices].clone()
    target = dataset.train.target[indices].clone()
    frame = dataset.train.frame.iloc[indices.detach().cpu().numpy()].reset_index(drop=True)

    if target_noise_std > 0.0:
        scale = dataset.train.target.std().clamp(min=1e-8)
        noise = torch.randn(target.shape, generator=gen, device=target.device) * target_noise_std * scale
        target = torch.clamp(target + noise, min=0.0)

    return SpatialDataset(
        train=SpatialSplit(features, target, frame),
        val=dataset.val,
        test=dataset.test,
        feature_columns=dataset.feature_columns,
    )


def _plot_rmse(summary: pd.DataFrame, output_path: Path) -> None:
    test = summary[summary["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in test.groupby("model"):
        grouped = group.groupby("train_fraction")["rmse"].mean().reset_index().sort_values("train_fraction")
        ax.plot(grouped["train_fraction"], grouped["rmse"], marker="o", label=model_name)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Training fraction kept")
    ax.set_ylabel("Test RMSE (mm/h)")
    ax.set_title("Spatial Data Efficiency: Does Physics+Spatial Help Under Sparse Data?")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_improvement(summary: pd.DataFrame, output_path: Path) -> None:
    test = summary[summary["split"] == "test"].copy()
    single = test[test["model"] == "single_link_nn"].groupby("train_fraction")["rmse"].mean()
    spatial = test[test["model"] == "spatial_physics_nn"].groupby("train_fraction")["rmse"].mean()
    merged = pd.DataFrame({"single": single, "spatial": spatial}).dropna()
    merged["improvement_pct"] = (merged["single"] - merged["spatial"]) / merged["single"] * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"{f:.0%}" for f in merged.index], merged["improvement_pct"])
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1, label="10% target")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Training fraction kept")
    ax.set_ylabel("RMSE improvement over single-link (%)")
    ax.set_title("Spatial+Physics Improvement vs Data Availability")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    rendered = frame.copy()
    for col in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[col]):
            rendered[col] = rendered[col].map(lambda v: f"{v:.6g}")
        else:
            rendered[col] = rendered[col].astype(str)
    cols = list(rendered.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_report(
    output_path: Path,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    test = summary[summary["split"] == "test"].copy()
    pivot = (
        test.groupby(["train_fraction", "model"])["rmse"]
        .mean()
        .reset_index()
        .pivot(index="train_fraction", columns="model", values="rmse")
        .reset_index()
        .sort_values("train_fraction", ascending=False)
    )
    if "single_link_nn" in pivot.columns and "spatial_physics_nn" in pivot.columns:
        pivot["spatial_improvement_pct"] = (
            (pivot["single_link_nn"] - pivot["spatial_physics_nn"])
            / pivot["single_link_nn"]
            * 100.0
        )
        wins = int((pivot["spatial_improvement_pct"] > 0).sum())
        above_10 = int((pivot["spatial_improvement_pct"] >= 10.0).sum())
        best_improvement = float(pivot["spatial_improvement_pct"].max())
    else:
        wins = 0
        above_10 = 0
        best_improvement = float("nan")

    if above_10 > 0:
        conclusion = (
            f"Spatial+physics beats single-link in {wins}/{len(pivot)} data fractions, "
            f"with {above_10} reaching the 10% improvement target. "
            f"Best improvement: {best_improvement:.3g}%."
        )
    elif wins > 0:
        conclusion = (
            f"Spatial+physics beats single-link in {wins}/{len(pivot)} data fractions, "
            f"but none reach 10%. Best improvement: {best_improvement:.3g}%."
        )
    else:
        conclusion = "Spatial+physics does not consistently beat single-link under sparse data."

    lines = [
        "# Spatial Robustness / Data-Efficiency Report",
        "",
        f"Input: `{args.input}`",
        f"Train fractions: `{args.train_fractions}`",
        f"Seeds: `{args.seeds}`",
        f"Target noise std: `{args.target_noise_std}`",
        f"Max links per target: `{args.max_links_per_target}`",
        f"Rain weight alpha: `{args.rain_weight_alpha}`",
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
        "- Each spatial+physics model is compared to a matched single-link baseline trained with the same seed, lr, and data fraction.",
        "- Training rows are randomly removed; validation and test stay clean.",
        "- Positive `spatial_improvement_pct` means spatial+physics was better than single-link at that data fraction.",
        "- This tests whether spatial CML context provides a data-efficiency advantage, directly addressing Metric 2.",
        "",
        "## Visual Artifacts",
        "",
        "- `rmse_vs_train_fraction.png`: test RMSE curves for all models.",
        "- `improvement_vs_fraction.png`: spatial improvement percentage at each data fraction.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fractions = _parse_float_list(args.train_fractions)
    seeds = _parse_int_list(args.seeds)

    wide, power_a = build_spatial_wide_table(args.input, args.power_b, args.max_links_per_target)

    modes = {
        "single_link_nn": "single_link",
        "single_link_physics_nn": "single_link_physics",
        "spatial_nn": "spatial",
        "spatial_physics_nn": "spatial_physics",
    }

    rows: list[dict[str, float | str]] = []

    for seed in seeds:
        for fraction in fractions:
            base_datasets = {}
            for model_name, mode in modes.items():
                base = prepare_spatial_dataset(wide, mode, args.max_links_per_target, args.device)
                limited = limit_spatial_training(base, fraction, seed, args.target_noise_std)
                base_datasets[model_name] = limited

            shared = {
                "epochs": args.epochs,
                "lr": args.lr,
                "seed": seed,
                "device": args.device,
                "patience": args.patience,
                "hidden_dims": [64, 64],
                "rain_weight_alpha": args.rain_weight_alpha,
                "wet_threshold_mm_h": args.wet_threshold_mm_h,
            }

            for model_name, dataset in base_datasets.items():
                model = train_spatial_model(dataset, **shared)
                metrics, _ = evaluate_spatial_model(
                    model_name, model, dataset, args.wet_threshold_mm_h,
                )
                for row in metrics:
                    row.update({
                        "train_fraction": fraction,
                        "seed": seed,
                        "power_a": power_a,
                    })
                    rows.append(row)

            single_test = [r for r in rows if r["model"] == "single_link_nn" and r["split"] == "test" and r["train_fraction"] == fraction and r["seed"] == seed]
            spatial_test = [r for r in rows if r["model"] == "spatial_physics_nn" and r["split"] == "test" and r["train_fraction"] == fraction and r["seed"] == seed]
            if single_test and spatial_test:
                s_rmse = float(single_test[-1]["rmse"])
                sp_rmse = float(spatial_test[-1]["rmse"])
                imp = (s_rmse - sp_rmse) / s_rmse * 100.0
                print(f"  seed={seed} frac={fraction:.0%}: single={s_rmse:.4f} spatial_physics={sp_rmse:.4f} improvement={imp:.1f}%")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "spatial_robustness_metrics.csv", index=False)
    _plot_rmse(summary, args.output_dir / "rmse_vs_train_fraction.png")
    _plot_improvement(summary, args.output_dir / "improvement_vs_fraction.png")
    _write_report(args.output_dir / "spatial_robustness_report.md", summary, args)

    test = summary[summary["split"] == "test"]
    print("\nSpatial robustness test RMSE")
    print("-" * 72)
    print(
        test.pivot_table(index="train_fraction", columns="model", values="rmse", aggfunc="mean")
        .sort_index(ascending=False)
        .to_string(float_format=lambda v: f"{v:.4f}")
    )
    print(f"\nSaved report: {args.output_dir / 'spatial_robustness_report.md'}")


if __name__ == "__main__":
    main()
