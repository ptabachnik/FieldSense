#!/usr/bin/env python3
"""
Build presentation-ready Phase 2/3 summary artifacts.

This script does not train models. It reads existing validation/model outputs
and creates concise figures plus a markdown summary for reports/slides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create presentation-ready FieldSense result artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "presentation")
    parser.add_argument("--phase2-report", type=Path, default=Path("outputs") / "phase2_validation" / "phase2_validation_report.md")
    parser.add_argument("--phase3-metrics", type=Path, default=Path("outputs") / "phase3b" / "metrics.csv")
    parser.add_argument("--spatial-metrics", type=Path, default=Path("outputs") / "phase3c_spatial_k2" / "spatial_metrics.csv")
    parser.add_argument("--sweep-results", type=Path, default=Path("outputs") / "phase3d_spatial_sweep" / "spatial_sweep_results.csv")
    parser.add_argument("--robustness-results", type=Path, default=Path("outputs") / "phase3_robustness_lambda001" / "robustness_metrics.csv")
    parser.add_argument("--wetdry-metrics", type=Path, default=Path("outputs") / "phase3e_wetdry" / "wetdry_metrics.csv")
    return parser.parse_args()


def load_phase3_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    return metrics[metrics["split"] == "test"].copy()


def load_spatial_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    return metrics[metrics["split"] == "test"].copy()


def load_sweep_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_wetdry_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    return metrics[metrics["split"] == "test"].copy()


def improvement_pct(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline * 100.0


def build_summary_tables(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    phase3 = load_phase3_metrics(args.phase3_metrics)
    spatial = load_spatial_metrics(args.spatial_metrics)
    sweep = load_sweep_results(args.sweep_results)
    wetdry = load_wetdry_metrics(args.wetdry_metrics)

    nn_rmse = float(phase3.loc[phase3["model"] == "nn", "rmse"].iloc[0])
    loss_pinn_rmse = float(phase3.loc[phase3["model"] == "pinn_lambda_0.01", "rmse"].iloc[0])
    feature_rmse = float(phase3.loc[phase3["model"] == "physics_feature_nn", "rmse"].iloc[0])

    single_rmse = float(spatial.loc[spatial["model"] == "single_link_nn", "rmse"].iloc[0])
    spatial_rmse = float(spatial.loc[spatial["model"] == "spatial_nn", "rmse"].iloc[0])
    single_f1 = float(spatial.loc[spatial["model"] == "single_link_nn", "wet_f1"].iloc[0])
    spatial_f1 = float(spatial.loc[spatial["model"] == "spatial_nn", "wet_f1"].iloc[0])

    best_improvement = sweep.sort_values("improvement_pct", ascending=False).iloc[0]
    best_rmse = sweep.sort_values("spatial_rmse").iloc[0]
    single_wetdry_f1 = float(wetdry.loc[wetdry["model"] == "single_link_classifier", "f1"].iloc[0])
    spatial_wetdry = wetdry.sort_values("f1", ascending=False).iloc[0]

    progress = pd.DataFrame(
        [
            {
                "stage": "3A rigid loss PINN",
                "baseline": "pure NN",
                "baseline_rmse": nn_rmse,
                "candidate_rmse": loss_pinn_rmse,
                "improvement_pct": improvement_pct(nn_rmse, loss_pinn_rmse),
                "claim": "Rigid loss over-constrains",
            },
            {
                "stage": "3B physics feature",
                "baseline": "pure NN",
                "baseline_rmse": nn_rmse,
                "candidate_rmse": feature_rmse,
                "improvement_pct": improvement_pct(nn_rmse, feature_rmse),
                "claim": "Physics as feature helps slightly",
            },
            {
                "stage": "3C spatial K=2",
                "baseline": "one-link NN",
                "baseline_rmse": single_rmse,
                "candidate_rmse": spatial_rmse,
                "improvement_pct": improvement_pct(single_rmse, spatial_rmse),
                "claim": "Spatial links help",
            },
            {
                "stage": "3D best absolute RMSE",
                "baseline": "matched one-link NN",
                "baseline_rmse": float(best_rmse["single_rmse"]),
                "candidate_rmse": float(best_rmse["spatial_rmse"]),
                "improvement_pct": float(best_rmse["improvement_pct"]),
                "claim": "10% target reached",
            },
            {
                "stage": "3D best improvement",
                "baseline": "matched one-link NN",
                "baseline_rmse": float(best_improvement["single_rmse"]),
                "candidate_rmse": float(best_improvement["spatial_rmse"]),
                "improvement_pct": float(best_improvement["improvement_pct"]),
                "claim": "Largest relative gain",
            },
        ]
    )

    wetdry_compare = pd.DataFrame(
        [
            {
                "metric": "Wet/Dry F1",
                "one_link": single_wetdry_f1,
                "spatial": float(spatial_wetdry["f1"]),
                "relative_gain_pct": (float(spatial_wetdry["f1"]) - single_wetdry_f1) / single_wetdry_f1 * 100.0,
            },
            {
                "metric": "Precision",
                "one_link": float(wetdry.loc[wetdry["model"] == "single_link_classifier", "precision"].iloc[0]),
                "spatial": float(spatial_wetdry["precision"]),
                "relative_gain_pct": (
                    float(spatial_wetdry["precision"])
                    - float(wetdry.loc[wetdry["model"] == "single_link_classifier", "precision"].iloc[0])
                )
                / float(wetdry.loc[wetdry["model"] == "single_link_classifier", "precision"].iloc[0])
                * 100.0,
            },
            {
                "metric": "Recall",
                "one_link": float(wetdry.loc[wetdry["model"] == "single_link_classifier", "recall"].iloc[0]),
                "spatial": float(spatial_wetdry["recall"]),
                "relative_gain_pct": (
                    float(spatial_wetdry["recall"])
                    - float(wetdry.loc[wetdry["model"] == "single_link_classifier", "recall"].iloc[0])
                )
                / float(wetdry.loc[wetdry["model"] == "single_link_classifier", "recall"].iloc[0])
                * 100.0,
            },
        ]
    )

    spatial_compare = pd.DataFrame(
        [
            {"metric": "RMSE", "one_link": single_rmse, "spatial": spatial_rmse, "relative_gain_pct": improvement_pct(single_rmse, spatial_rmse)},
            {"metric": "Rainy RMSE", "one_link": float(spatial.loc[spatial["model"] == "single_link_nn", "rainy_rmse"].iloc[0]), "spatial": float(spatial.loc[spatial["model"] == "spatial_nn", "rainy_rmse"].iloc[0]), "relative_gain_pct": improvement_pct(float(spatial.loc[spatial["model"] == "single_link_nn", "rainy_rmse"].iloc[0]), float(spatial.loc[spatial["model"] == "spatial_nn", "rainy_rmse"].iloc[0]))},
            {"metric": "Wet/Dry F1", "one_link": single_f1, "spatial": spatial_f1, "relative_gain_pct": (spatial_f1 - single_f1) / single_f1 * 100.0},
        ]
    )

    return {
        "progress": progress,
        "spatial_compare": spatial_compare,
        "phase3": phase3,
        "spatial": spatial,
        "sweep": sweep,
        "wetdry": wetdry,
        "wetdry_compare": wetdry_compare,
    }


def plot_progress(progress: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = progress["stage"].tolist()
    x = range(len(progress))
    ax.bar([i - 0.18 for i in x], progress["baseline_rmse"], width=0.36, label="baseline")
    ax.bar([i + 0.18 for i in x], progress["candidate_rmse"], width=0.36, label="candidate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Test RMSE (mm/h)")
    ax.set_title("Model Progress Toward Project Goal")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_improvements(progress: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:red" if value < 0 else "tab:blue" for value in progress["improvement_pct"]]
    ax.bar(progress["stage"], progress["improvement_pct"], color=colors)
    ax.axhline(10.0, color="black", linestyle="--", linewidth=1, label="10% target")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Improvement over matched baseline (%)")
    ax.set_title("Improvement by Modeling Stage")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_spatial_metrics(spatial_compare: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (_, row) in zip(axes, spatial_compare.iterrows()):
        ax.bar(["one-link", "spatial"], [row["one_link"], row["spatial"]])
        ax.set_title(f"{row['metric']}\nGain {row['relative_gain_pct']:.1f}%")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_wetdry_metrics(wetdry: pd.DataFrame, output_path: Path) -> None:
    test = wetdry.sort_values("f1", ascending=False)
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
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_report(output_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    progress = tables["progress"].copy()
    spatial_compare = tables["spatial_compare"].copy()
    wetdry = tables["wetdry"].copy().sort_values("f1", ascending=False)
    wetdry_compare = tables["wetdry_compare"].copy()
    best_abs = progress[progress["stage"] == "3D best absolute RMSE"].iloc[0]
    best_improvement = progress[progress["stage"] == "3D best improvement"].iloc[0]

    lines = [
        "# FieldSense Presentation Summary",
        "",
        "## Core Claim",
        "",
        "Physics as a rigid power-law loss is too idealized for real CML/gauge data, but physics-derived features and spatial CML context improve rain-rate estimation.",
        "",
        "## Goal Progress",
        "",
        dataframe_to_markdown(progress),
        "",
        "## Spatial Improvement",
        "",
        dataframe_to_markdown(spatial_compare),
        "",
        "## Wet/Dry Classification",
        "",
        dataframe_to_markdown(wetdry[["model", "f1", "precision", "recall", "accuracy"]]),
        "",
        "## Wet/Dry Improvement",
        "",
        dataframe_to_markdown(wetdry_compare),
        "",
        "## Best Results",
        "",
        f"- Best absolute Phase 3D spatial RMSE: `{best_abs['candidate_rmse']:.6g}` with `{best_abs['improvement_pct']:.3g}%` improvement over matched one-link baseline.",
        f"- Largest Phase 3D relative gain: `{best_improvement['improvement_pct']:.3g}%` improvement.",
        f"- Best wet/dry classifier: `{wetdry.iloc[0]['model']}` with F1 `{wetdry.iloc[0]['f1']:.3g}`.",
        "- Phase 3A rigid-loss PINN did not improve over the NN, supporting the interpretation that power law is useful as a prior/feature, not as a hard constraint.",
        "",
        "## Slide-Ready Figures",
        "",
        "- `model_progress_rmse.png`",
        "- `improvement_by_stage.png`",
        "- `spatial_metric_comparison.png`",
        "- `wetdry_metric_comparison.png`",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render markdown without optional tabulate dependency."""
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_summary_tables(args)
    tables["progress"].to_csv(args.output_dir / "goal_progress_table.csv", index=False)
    tables["spatial_compare"].to_csv(args.output_dir / "spatial_comparison_table.csv", index=False)
    plot_progress(tables["progress"], args.output_dir / "model_progress_rmse.png")
    plot_improvements(tables["progress"], args.output_dir / "improvement_by_stage.png")
    plot_spatial_metrics(tables["spatial_compare"], args.output_dir / "spatial_metric_comparison.png")
    plot_wetdry_metrics(tables["wetdry"], args.output_dir / "wetdry_metric_comparison.png")
    write_summary_report(args.output_dir / "presentation_summary.md", tables)

    print(f"Presentation summary: {args.output_dir / 'presentation_summary.md'}")
    print(f"Figures: {args.output_dir}")


if __name__ == "__main__":
    main()
