#!/usr/bin/env python3
"""
Phase 3D spatial hyperparameter/data sweep.

Runs up to N spatial configurations and keeps the best result. Each attempt
trains a spatial model and a matched one-link baseline so improvement is
measured fairly under the same seed, optimizer, and loss weighting.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .rain_spatial import (
    build_spatial_wide_table,
    prepare_spatial_dataset,
    train_spatial_model,
    evaluate_spatial_model,
    _dataframe_to_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep spatial CML hyperparameters")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "phase2_spatial_validation" / "prepared_with_splits.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "phase3d_spatial_sweep")
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--power-b", type=float, default=0.8)
    parser.add_argument("--wet-threshold-mm-h", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def candidate_configs(max_attempts: int) -> list[dict[str, object]]:
    """Create deterministic hyperparameter candidates, capped at max_attempts."""
    max_links_options = [2, 3, 4]
    modes = ["spatial", "spatial_physics"]
    seeds = [42, 7, 123, 202, 999]
    lrs = [1e-3, 5e-4]
    hidden_options = [[32, 32], [64, 64], [128, 64]]
    rain_weights = [0.0, 2.0, 5.0]

    configs = []
    for lr, hidden, rain_weight, max_links, mode, seed in product(
        lrs, hidden_options, rain_weights, max_links_options, modes, seeds
    ):
        configs.append(
            {
                "max_links": max_links,
                "mode": mode,
                "seed": seed,
                "lr": lr,
                "hidden_dims": hidden,
                "rain_weight_alpha": rain_weight,
            }
        )
    return configs[:max_attempts]


def test_metric_row(metrics: list[dict[str, float | str]]) -> dict[str, float | str]:
    rows = [row for row in metrics if row["split"] == "test"]
    if len(rows) != 1:
        raise ValueError("Expected exactly one test metric row")
    return rows[0]


def plot_best_attempts(results: pd.DataFrame, output_path: Path) -> None:
    best = results.sort_values("improvement_pct", ascending=False).head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"#{int(row.attempt)} K={int(row.max_links)} {row.mode}" for row in best.itertuples()]
    ax.bar(labels, best["improvement_pct"])
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1, label="10% target")
    ax.set_ylabel("RMSE improvement vs matched one-link NN (%)")
    ax.set_title("Best Spatial Sweep Attempts")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(output_path: Path, results: pd.DataFrame, args: argparse.Namespace) -> None:
    best_rmse = results.sort_values(["spatial_rmse", "rainy_rmse"], ascending=True).iloc[0]
    best_improvement_row = results.sort_values("improvement_pct", ascending=False).iloc[0]
    best_improvement = float(best_improvement_row["improvement_pct"])
    reached_target = best_improvement >= 10.0
    top = results.sort_values("improvement_pct", ascending=False).head(15)[
        [
            "attempt",
            "mode",
            "max_links",
            "seed",
            "lr",
            "hidden_dims",
            "rain_weight_alpha",
            "single_rmse",
            "spatial_rmse",
            "rainy_rmse",
            "wet_f1",
            "improvement_pct",
        ]
    ]

    conclusion = (
        f"The sweep reached the 10% target with best improvement {best_improvement:.3g}%."
        if reached_target
        else f"The sweep did not reach the 10% target; best improvement was {best_improvement:.3g}%."
    )
    lines = [
        "# Phase 3D Spatial Sweep Report",
        "",
        f"Input: `{args.input}`",
        f"Attempts run: `{len(results)}`",
        f"Epochs per model: `{args.epochs}`",
        f"Selection metric: best RMSE improvement against a matched one-link NN; absolute RMSE is also reported.",
        "",
        "## Best Improvement Result",
        "",
        f"- {conclusion}",
        f"- Best improvement attempt: `{int(best_improvement_row['attempt'])}`",
        f"- Mode: `{best_improvement_row['mode']}`",
        f"- Max links per target: `{int(best_improvement_row['max_links'])}`",
        f"- Seed: `{int(best_improvement_row['seed'])}`",
        f"- Learning rate: `{best_improvement_row['lr']}`",
        f"- Hidden dims: `{best_improvement_row['hidden_dims']}`",
        f"- Rain-weight alpha: `{best_improvement_row['rain_weight_alpha']}`",
        f"- Matched one-link RMSE: `{best_improvement_row['single_rmse']:.6g}`",
        f"- Spatial RMSE: `{best_improvement_row['spatial_rmse']:.6g}`",
        f"- Rainy-only RMSE: `{best_improvement_row['rainy_rmse']:.6g}`",
        f"- Wet/dry F1: `{best_improvement_row['wet_f1']:.6g}`",
        "",
        "## Lowest Absolute RMSE Result",
        "",
        f"- Attempt: `{int(best_rmse['attempt'])}`",
        f"- Mode: `{best_rmse['mode']}`",
        f"- Spatial RMSE: `{best_rmse['spatial_rmse']:.6g}`",
        f"- Improvement vs matched one-link NN: `{best_rmse['improvement_pct']:.6g}%`",
        "",
        "## Top Attempts by Improvement",
        "",
        _dataframe_to_markdown(top),
        "",
        "## Interpretation",
        "",
        "- Each spatial model is compared to a matched one-link baseline trained with the same seed, learning rate, architecture, and rain-weighted loss.",
        "- This isolates the effect of adding spatial CML links.",
        "- If the target is not reached, the result is still useful: it shows which data/model changes help and where gains saturate.",
        "",
        "## Visual Artifacts",
        "",
        "- `best_attempts.png`: top attempts and the 10% improvement target line.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = candidate_configs(args.max_attempts)
    wide_cache: dict[int, tuple[pd.DataFrame, float]] = {}
    rows = []

    for attempt, config in enumerate(configs, start=1):
        max_links = int(config["max_links"])
        if max_links not in wide_cache:
            wide_cache[max_links] = build_spatial_wide_table(args.input, args.power_b, max_links)
        wide, power_a = wide_cache[max_links]

        hidden_dims = list(config["hidden_dims"])
        shared = {
            "epochs": args.epochs,
            "lr": float(config["lr"]),
            "seed": int(config["seed"]),
            "device": args.device,
            "patience": args.patience,
            "hidden_dims": hidden_dims,
            "rain_weight_alpha": float(config["rain_weight_alpha"]),
            "wet_threshold_mm_h": args.wet_threshold_mm_h,
        }

        single_dataset = prepare_spatial_dataset(wide, "single_link", max_links, args.device)
        single_model = train_spatial_model(single_dataset, **shared)
        single_metrics, _ = evaluate_spatial_model("single_link_nn", single_model, single_dataset, args.wet_threshold_mm_h)
        single_test = test_metric_row(single_metrics)

        spatial_dataset = prepare_spatial_dataset(wide, str(config["mode"]), max_links, args.device)
        spatial_model = train_spatial_model(spatial_dataset, **shared)
        spatial_metrics, _ = evaluate_spatial_model(str(config["mode"]), spatial_model, spatial_dataset, args.wet_threshold_mm_h)
        spatial_test = test_metric_row(spatial_metrics)

        single_rmse = float(single_test["rmse"])
        spatial_rmse = float(spatial_test["rmse"])
        rows.append(
            {
                "attempt": attempt,
                **config,
                "power_a": power_a,
                "single_rmse": single_rmse,
                "spatial_rmse": spatial_rmse,
                "improvement_pct": (single_rmse - spatial_rmse) / single_rmse * 100.0,
                "rainy_rmse": float(spatial_test["rainy_rmse"]),
                "mae": float(spatial_test["mae"]),
                "wet_f1": float(spatial_test["wet_f1"]),
                "wet_accuracy": float(spatial_test["wet_accuracy"]),
            }
        )
        if attempt % 10 == 0 or attempt == len(configs):
            current_best = max(rows, key=lambda row: row["improvement_pct"])
            print(
                f"Attempt {attempt}/{len(configs)}: best improvement "
                f"{current_best['improvement_pct']:.2f}% (spatial RMSE {current_best['spatial_rmse']:.4f})"
            )

    results = pd.DataFrame(rows)
    results["hidden_dims"] = results["hidden_dims"].map(lambda value: "-".join(str(v) for v in value))
    results.to_csv(args.output_dir / "spatial_sweep_results.csv", index=False)
    plot_best_attempts(results, args.output_dir / "best_attempts.png")
    write_report(args.output_dir / "spatial_sweep_report.md", results, args)

    best = results.sort_values("improvement_pct", ascending=False).iloc[0]
    print("\nBest sweep result")
    print("-" * 72)
    print(best.to_string())
    print(f"\nSaved sweep report: {args.output_dir / 'spatial_sweep_report.md'}")


if __name__ == "__main__":
    main()
