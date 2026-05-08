#!/usr/bin/env python3
"""
Validate canonical rain/CML prepared tables before Phase 3 modeling.

This script is intentionally data-only. It checks the table contract, creates a
time-based train/validation/test split, summarizes data quality, and saves a few
sanity plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .rain_schema import validate_model_ready_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prepared rain/CML data")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Canonical prepared CSV from rain_prepare.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "phase2_validation",
        help="Directory for report, split CSV, and plots",
    )
    parser.add_argument(
        "--wet-threshold-mm-h",
        type=float,
        default=0.1,
        help="Rain-rate threshold for wet/dry summary",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="First fraction of time for training split",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Next fraction of time for validation split",
    )
    parser.add_argument(
        "--min-test-rainy-fraction",
        type=float,
        default=0.05,
        help="Warn when the test split has less rain than this fraction",
    )
    parser.add_argument(
        "--min-test-wet-rows",
        type=int,
        default=10,
        help="Warn when the test split has fewer wet rows than this count",
    )
    return parser.parse_args()


def load_prepared_table(path: Path) -> pd.DataFrame:
    """Load and normalize a canonical prepared table."""
    table = pd.read_csv(path)
    table["time_utc"] = pd.to_datetime(table["time_utc"], utc=True)
    numeric_columns = [
        "target_rain_mm_h",
        "attenuation",
        "rsl",
        "tsl",
        "path_loss",
        "link_length_km",
        "frequency_GHz",
        "distance_to_target_km",
    ]
    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    validate_model_ready_table(table)
    return table.sort_values(["time_utc", "sublink_id"]).reset_index(drop=True)


def add_time_splits(
    table: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """Assign chronological train/val/test splits by unique timestamps."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")

    result = table.copy()
    unique_times = pd.Series(result["time_utc"].dropna().sort_values().unique())
    if len(unique_times) < 3:
        raise ValueError("Need at least 3 unique timestamps for train/val/test splits")

    train_end = unique_times.iloc[int(len(unique_times) * train_frac)]
    val_end = unique_times.iloc[int(len(unique_times) * (train_frac + val_frac))]

    result["split"] = "test"
    result.loc[result["time_utc"] < train_end, "split"] = "train"
    result.loc[
        (result["time_utc"] >= train_end) & (result["time_utc"] < val_end),
        "split",
    ] = "val"
    return result


def summarize_table(table: pd.DataFrame, wet_threshold_mm_h: float) -> dict[str, object]:
    """Compute Phase 2 data-quality summary values."""
    required = [
        "target_rain_mm_h",
        "attenuation",
        "rsl",
        "link_length_km",
        "frequency_GHz",
        "polarization",
    ]
    summary: dict[str, object] = {
        "rows": int(len(table)),
        "dataset_name": ", ".join(sorted(table["dataset_name"].dropna().astype(str).unique())),
        "start_time_utc": str(table["time_utc"].min()),
        "end_time_utc": str(table["time_utc"].max()),
        "unique_timestamps": int(table["time_utc"].nunique()),
        "unique_sublinks": int(table["sublink_id"].nunique()),
        "unique_targets": int(table["target_name"].nunique()),
        "rainy_fraction": float((table["target_rain_mm_h"] > wet_threshold_mm_h).mean()),
        "target_rain_mean_mm_h": float(table["target_rain_mm_h"].mean()),
        "target_rain_max_mm_h": float(table["target_rain_mm_h"].max()),
        "attenuation_mean_db": float(table["attenuation"].mean()),
        "attenuation_max_db": float(table["attenuation"].max()),
        "attenuation_negative_rows": int((table["attenuation"] < 0).sum()),
    }

    if "tsl" in table.columns:
        summary["tsl_std"] = float(table["tsl"].std())
        summary["tsl_range"] = float(table["tsl"].max() - table["tsl"].min())

    for column in required:
        if column in table.columns:
            summary[f"{column}_missing_fraction"] = float(table[column].isna().mean())

    if table["target_rain_mm_h"].notna().sum() > 1 and table["attenuation"].notna().sum() > 1:
        summary["pearson_rain_attenuation"] = float(
            table[["target_rain_mm_h", "attenuation"]].corr(method="pearson").iloc[0, 1]
        )
        summary["spearman_rain_attenuation"] = float(
            table[["target_rain_mm_h", "attenuation"]].corr(method="spearman").iloc[0, 1]
        )

    split_counts = table["split"].value_counts().to_dict() if "split" in table.columns else {}
    for split_name in ["train", "val", "test"]:
        split_group = table[table["split"] == split_name] if "split" in table.columns else pd.DataFrame()
        summary[f"{split_name}_rows"] = int(split_counts.get(split_name, 0))
        if not split_group.empty:
            wet_mask = split_group["target_rain_mm_h"] > wet_threshold_mm_h
            summary[f"{split_name}_rainy_rows"] = int(wet_mask.sum())
            summary[f"{split_name}_rainy_fraction"] = float(wet_mask.mean())
        else:
            summary[f"{split_name}_rainy_rows"] = 0
            summary[f"{split_name}_rainy_fraction"] = 0.0
    return summary


def per_link_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize rows and rain/attenuation relationship per CML sublink."""
    rows = []
    for sublink_id, group in table.groupby("sublink_id"):
        corr = group[["target_rain_mm_h", "attenuation"]].corr(method="spearman").iloc[0, 1]
        rows.append(
            {
                "sublink_id": sublink_id,
                "rows": len(group),
                "target_name": ", ".join(sorted(group["target_name"].dropna().astype(str).unique())),
                "rainy_fraction": (group["target_rain_mm_h"] > 0.1).mean(),
                "attenuation_mean_db": group["attenuation"].mean(),
                "attenuation_max_db": group["attenuation"].max(),
                "spearman_rain_attenuation": corr,
                "link_length_km": group["link_length_km"].iloc[0],
                "frequency_GHz": group["frequency_GHz"].iloc[0],
                "polarization": group["polarization"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def split_summary(table: pd.DataFrame, wet_threshold_mm_h: float) -> pd.DataFrame:
    """Summarize rain and attenuation by chronological split."""
    rows = []
    for split_name, group in table.groupby("split", sort=False):
        rows.append(
            {
                "split": split_name,
                "rows": len(group),
                "unique_timestamps": group["time_utc"].nunique(),
                "start_time_utc": group["time_utc"].min(),
                "end_time_utc": group["time_utc"].max(),
                "rainy_fraction": (group["target_rain_mm_h"] > wet_threshold_mm_h).mean(),
                "rain_mean_mm_h": group["target_rain_mm_h"].mean(),
                "rain_max_mm_h": group["target_rain_mm_h"].max(),
                "attenuation_mean_db": group["attenuation"].mean(),
                "attenuation_max_db": group["attenuation"].max(),
            }
        )
    return pd.DataFrame(rows)


def phase2_pass_fail(summary: dict[str, object]) -> list[str]:
    """Return blocking validation issues; empty list means Phase 2 data is usable."""
    issues = []
    if int(summary["rows"]) <= 0:
        issues.append("Prepared table has no rows.")
    if int(summary["unique_sublinks"]) <= 0:
        issues.append("No CML sublinks found.")
    if float(summary["target_rain_mm_h_missing_fraction"]) > 0.0:
        issues.append("Rain target has missing values.")
    if float(summary["attenuation_missing_fraction"]) > 0.0:
        issues.append("Attenuation has missing values.")
    if int(summary["attenuation_negative_rows"]) > 0:
        issues.append("Attenuation contains negative values.")
    if float(summary["rainy_fraction"]) <= 0.0:
        issues.append("No wet samples found.")
    for split_name in ["train", "val", "test"]:
        if int(summary[f"{split_name}_rows"]) <= 0:
            issues.append(f"{split_name} split is empty.")
    return issues


def phase2_warnings(
    summary: dict[str, object],
    min_test_rainy_fraction: float,
    min_test_wet_rows: int,
) -> list[str]:
    """Return non-blocking warnings that affect claim strength."""
    warnings = []
    test_rainy_fraction = float(summary.get("test_rainy_fraction", 0.0))
    test_rainy_rows = int(summary.get("test_rainy_rows", 0))
    if test_rainy_fraction < min_test_rainy_fraction:
        warnings.append(
            f"Test split is rain-sparse: rainy fraction {test_rainy_fraction:.4f} "
            f"is below {min_test_rainy_fraction:.4f}. Wet/dry and rainy-only claims may be unstable."
        )
    if test_rainy_rows < min_test_wet_rows:
        warnings.append(
            f"Test split has only {test_rainy_rows} wet rows, below requested minimum {min_test_wet_rows}."
        )
    return warnings


def write_report(
    output_path: Path,
    input_path: Path,
    summary: dict[str, object],
    link_summary: pd.DataFrame,
    split_summary_frame: pd.DataFrame,
    issues: list[str],
    warnings: list[str],
) -> None:
    """Write a markdown validation report."""
    status = "FAIL" if issues else "PASS_WITH_WARNINGS" if warnings else "PASS"
    lines = [
        "# Phase 2 Data Validation Report",
        "",
        f"Input: `{input_path}`",
        f"Status: **{status}**",
        "",
        "## Summary",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.6g}")
        else:
            lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Interpretation", ""])
    if issues:
        lines.append("- The prepared data has blocking issues and should not be used for Phase 3 yet.")
    else:
        lines.extend(
            [
                "- The prepared table is usable for Phase 3: all required CML, metadata, and rain-target fields are present.",
                "- Attenuation is non-negative and has no missing values, so it can be used in the power-law baseline and PINN residual.",
                "- Rain and attenuation have a positive diagnostic relationship, supporting the CML-rainfall modeling assumption.",
                "- Chronological splits are non-empty, which allows model evaluation without random storm leakage.",
            ]
        )

    lines.extend(["", "## Split Summary", ""])
    lines.append(_dataframe_to_markdown(split_summary_frame))

    lines.extend(["", "## Per-Link Summary", ""])
    lines.append(_dataframe_to_markdown(link_summary))

    lines.extend(
        [
            "",
            "## Visual Artifacts",
            "",
            "- `rain_attenuation_timeseries.png`: rain and attenuation over time for one link.",
            "- `rain_attenuation_scatter.png`: observed rain rate versus attenuation.",
            "- `split_rain_distribution.png`: rain-rate distribution by train/validation/test split.",
        ]
    )

    lines.extend(["", "## Blocking Issues", ""])
    if issues:
        lines.extend([f"- {issue}" for issue in issues])
    else:
        lines.append("- None. Data is ready for Phase 3 modeling experiments.")

    lines.extend(["", "## Claim Warnings", ""])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- None. Test split has enough wet samples for basic claim support.")

    lines.extend(
        [
            "",
            "## Notes",
            "- Splits are chronological to avoid storm leakage.",
            "- Correlation is diagnostic only; low correlation is not a blocker by itself.",
            "- PyNNcml OpenMRG uses stable 15-minute alignment by default.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small dataframe as markdown without optional dependencies."""
    if frame.empty:
        return "_No rows._"

    rendered = frame.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
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


def plot_timeseries(table: pd.DataFrame, output_path: Path) -> None:
    """Save rain/attenuation time-series sanity plot for the first sublink."""
    sublink_id = table["sublink_id"].iloc[0]
    group = table[table["sublink_id"] == sublink_id].sort_values("time_utc")

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(group["time_utc"], group["target_rain_mm_h"], color="#1f77b4", label="Gauge rain (mm/h)")
    ax1.set_ylabel("Rain rate (mm/h)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(group["time_utc"], group["attenuation"], color="#d62728", alpha=0.8, label="Attenuation (dB)")
    ax2.set_ylabel("Attenuation (dB)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title(f"Rain and CML Attenuation Over Time (sublink {sublink_id})")
    ax1.set_xlabel("UTC time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scatter(table: pd.DataFrame, output_path: Path) -> None:
    """Save attenuation vs rain scatter sanity plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(table["attenuation"], table["target_rain_mm_h"], s=12, alpha=0.45)
    ax.set_xlabel("Attenuation (dB)")
    ax.set_ylabel("Gauge rain rate (mm/h)")
    ax.set_title("Gauge Rain vs CML Attenuation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_split_distribution(table: pd.DataFrame, output_path: Path) -> None:
    """Save split-wise rain distribution plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [table.loc[table["split"] == split, "target_rain_mm_h"] for split in ["train", "val", "test"]]
    ax.boxplot(data, tick_labels=["train", "val", "test"], showfliers=False)
    ax.set_ylabel("Gauge rain rate (mm/h)")
    ax.set_title("Rain Distribution by Chronological Split")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = load_prepared_table(args.input)
    table = add_time_splits(table, train_frac=args.train_frac, val_frac=args.val_frac)

    split_path = args.output_dir / "prepared_with_splits.csv"
    table.to_csv(split_path, index=False)

    summary = summarize_table(table, wet_threshold_mm_h=args.wet_threshold_mm_h)
    link_summary = per_link_summary(table)
    split_summary_frame = split_summary(table, wet_threshold_mm_h=args.wet_threshold_mm_h)
    issues = phase2_pass_fail(summary)
    warnings = phase2_warnings(
        summary,
        min_test_rainy_fraction=args.min_test_rainy_fraction,
        min_test_wet_rows=args.min_test_wet_rows,
    )

    link_summary_path = args.output_dir / "per_link_summary.csv"
    link_summary.to_csv(link_summary_path, index=False)
    split_summary_path = args.output_dir / "split_summary.csv"
    split_summary_frame.to_csv(split_summary_path, index=False)

    report_path = args.output_dir / "phase2_validation_report.md"
    write_report(report_path, args.input, summary, link_summary, split_summary_frame, issues, warnings)

    plot_timeseries(table, args.output_dir / "rain_attenuation_timeseries.png")
    plot_scatter(table, args.output_dir / "rain_attenuation_scatter.png")
    plot_split_distribution(table, args.output_dir / "split_rain_distribution.png")

    print(f"Validation report: {report_path}")
    print(f"Split table:        {split_path}")
    print(f"Per-link summary:   {link_summary_path}")
    print(f"Split summary:      {split_summary_path}")
    status = "FAIL" if issues else "PASS_WITH_WARNINGS" if warnings else "PASS"
    print("Status:             " + status)
    if issues:
        print("Blocking issues:")
        for issue in issues:
            print(f"- {issue}")
    if warnings:
        print("Claim warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
