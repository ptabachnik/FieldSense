#!/usr/bin/env python3
"""
Prepare canonical rain/CML tables for Phase 3 modeling.

OpenMRG can be prepared either from repo-local files or through PyNNcml's
dataset loader when the repo copy is missing cml.nc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .rain_data import build_stage1_table, describe_stage1_table
from .rain_pynncml import build_pynncml_openmrg_table
from .rain_schema import validate_model_ready_table, write_prepared_table


SUPPORTED_DATASETS = ("openmrg", "openmrg_sweden", "pynncml_openmrg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare rain/CML data for modeling")
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="openmrg",
        help="Dataset adapter to use.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root. For OpenMRG: dataset/open_datasets/OpenMRG_Sweden",
    )
    parser.add_argument(
        "--gauge",
        default="Chalm",
        help="Target gauge/station for the first prepared table",
    )
    parser.add_argument(
        "--sublink-id",
        type=int,
        default=None,
        help="Optional exact CML sublink. If omitted, nearest link(s) are selected.",
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=1,
        help="Number of nearest CML sublinks to include",
    )
    parser.add_argument(
        "--attenuation-method",
        choices=["rsl", "path_loss"],
        default="rsl",
        help="Stage 1 attenuation method",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path for the prepared canonical table",
    )
    parser.add_argument(
        "--time-start",
        default=None,
        help="Optional start time for PyNNcml/OpenMRG, e.g. 2015-06-01",
    )
    parser.add_argument(
        "--time-end",
        default=None,
        help="Optional end time for PyNNcml/OpenMRG, e.g. 2015-06-03",
    )
    parser.add_argument(
        "--link-distance-m",
        type=int,
        default=2000,
        help="Maximum PyNNcml link-to-gauge distance in meters",
    )
    parser.add_argument(
        "--rain-gauge-time-base-s",
        type=int,
        default=900,
        help="PyNNcml rain-gauge time base in seconds. Default keeps PyNNcml's stable 15-minute alignment.",
    )
    parser.add_argument(
        "--window-size-min",
        type=int,
        default=15,
        help="PyNNcml gauge rain-rate window in minutes.",
    )
    parser.add_argument(
        "--baseline-train-frac",
        type=float,
        default=0.7,
        help="Initial chronological fraction used to fit attenuation baselines.",
    )
    return parser.parse_args()


def _default_output_path(dataset: str, gauge: str) -> Path:
    safe_gauge = gauge.lower().replace(" ", "_")
    return Path("outputs") / "prepared" / f"{dataset}_stage1_{safe_gauge}.csv"


def _print_summary(summary: dict[str, float | int | str]) -> None:
    print("\nPrepared table summary")
    print("-" * 72)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:<32} {value:.6g}")
        else:
            print(f"{key:<32} {value}")


def main() -> None:
    args = parse_args()
    output = args.output or _default_output_path(args.dataset, args.gauge)

    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.dataset in ("openmrg", "openmrg_sweden"):
        table = build_stage1_table(
            root=args.data_root,
            gauge_name=args.gauge,
            sublink_id=args.sublink_id,
            max_links=args.max_links,
            attenuation_method=args.attenuation_method,
        )
    elif args.dataset == "pynncml_openmrg":
        table = build_pynncml_openmrg_table(
            data_root=args.data_root,
            time_start=args.time_start,
            time_end=args.time_end,
            max_links=args.max_links,
            link2gauge_distance_m=args.link_distance_m,
            rain_gauge_time_base_s=args.rain_gauge_time_base_s,
            window_size_in_min=args.window_size_min,
            baseline_train_fraction=args.baseline_train_frac,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    validate_model_ready_table(table)
    _print_summary(describe_stage1_table(table))

    write_prepared_table(table, output)
    print(f"\nSaved canonical prepared table: {output}")


if __name__ == "__main__":
    main()
