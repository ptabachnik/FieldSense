#!/usr/bin/env python3
"""
Inspect OpenMRG Sweden data readiness for Phase 2.

Run from projects/physics_ml:
    python -m src.phase2.rain_inspect

The default command is read-only: it reports available files, gauges, nearest
CML links, and whether cml.nc is present. Use --build-table after cml.nc exists
to verify the one-gauge/one-link aligned table.
"""

from __future__ import annotations

import argparse 
from pathlib import Path

from .rain_data import (
    available_city_gauges,
    build_stage1_table,
    check_data_availability,
    describe_stage1_table,
    find_nearest_links_to_gauge,
    load_city_gauge_metadata,
    load_cml_metadata,
    resolve_openmrg_paths,
)
from .rain_schema import write_prepared_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect OpenMRG Stage 1 data readiness")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to dataset/open_datasets/OpenMRG_Sweden",
    )
    parser.add_argument(
        "--gauge",
        default="Chalm",
        help="City gauge name to inspect first, e.g. Chalm or Drakeg",
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=5,
        help="Number of nearest CML sublinks to show",
    )
    parser.add_argument(
        "--build-table",
        action="store_true",
        help="Build the Stage 1 aligned table. Requires cml.nc.",
    )
    parser.add_argument(
        "--attenuation-method",
        choices=["rsl", "path_loss"],
        default="rsl",
        help="Simple Stage 1 attenuation method",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for the Stage 1 table when --build-table is used",
    )
    return parser.parse_args()


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _print_availability(availability: dict[str, bool]) -> None:
    _print_header("OpenMRG File Availability")
    for name, exists in availability.items():
        status = "OK" if exists else "MISSING"
        print(f"{name:<24} {status}")


def _print_nearest_links(gauge: str, nearest) -> None:
    _print_header(f"Nearest CML Sublinks to Gauge: {gauge}")
    columns = [
        "Link",
        "Direction",
        "Sublink",
        "distance_to_gauge_km",
        "Length_km",
        "Frequency_GHz",
        "Polarization",
    ]
    print(nearest[columns].to_string(index=False))


def _print_summary(summary: dict[str, float | int | str]) -> None:
    _print_header("Stage 1 Table Summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:<32} {value:.6g}")
        else:
            print(f"{key:<32} {value}")


def main() -> None:
    args = parse_args()
    paths = resolve_openmrg_paths(args.data_root)
    availability = check_data_availability(paths)

    print(f"OpenMRG root: {paths.root}")
    _print_availability(availability)

    if not availability["city_gauges"] or not availability["city_gauge_metadata"]:
        print("\nCity gauge files are missing; cannot continue Stage 1 inspection.")
        return

    gauges = available_city_gauges(paths)
    _print_header("Available City Gauges")
    print(", ".join(gauges))

    if not availability["cml_metadata"]:
        print("\nCML metadata is missing; cannot rank gauge/link distances.")
        return

    gauge_metadata = load_city_gauge_metadata(paths)
    cml_metadata = load_cml_metadata(paths)
    nearest = find_nearest_links_to_gauge(
        args.gauge,
        gauge_metadata,
        cml_metadata,
        max_links=args.max_links,
    )
    _print_nearest_links(args.gauge, nearest)

    if not availability["cml_nc"]:
        print(
            "\nNext step: restore or download cml.nc before building the aligned "
            "RSL/TSL + gauge table."
        )
        return

    if not args.build_table:
        print("\ncml.nc is present. Re-run with --build-table to verify alignment.")
        return

    table = build_stage1_table(
        root=args.data_root,
        gauge_name=args.gauge,
        max_links=1,
        attenuation_method=args.attenuation_method,
    )
    _print_summary(describe_stage1_table(table))

    if args.output is not None:
        write_prepared_table(table, args.output)
        print(f"\nSaved Stage 1 table: {args.output}")


if __name__ == "__main__":
    main()
