"""
Canonical prepared-table schema for rain/CML modeling.

Dataset adapters should output these columns so Phase 3 modeling can train on a
single table format regardless of whether the source is OpenMRG, OpenMesh, or a
future dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CANONICAL_RAIN_COLUMNS = [
    "dataset_name",
    "time_utc",
    "target_type",
    "target_name",
    "target_rain_mm_h",
    "target_rain_mm",
    "link_id",
    "sublink_id",
    "direction",
    "rsl",
    "tsl",
    "path_loss",
    "attenuation",
    "attenuation_method",
    "attenuation_baseline",
    "link_length_km",
    "frequency_GHz",
    "polarization",
    "distance_to_target_km",
    "gauge_name",
    "gauge_rain_mm",
    "gauge_rain_mm_h",
]

REQUIRED_MODEL_COLUMNS = [
    "dataset_name",
    "time_utc",
    "target_name",
    "target_rain_mm_h",
    "sublink_id",
    "rsl",
    "tsl",
    "path_loss",
    "attenuation",
    "link_length_km",
    "frequency_GHz",
    "polarization",
]


def canonicalize_rain_table(table: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Normalize an adapter output to the Phase 3 table contract.

    The OpenMRG adapter still keeps gauge-specific columns for readability, but
    the modeling path should consume the generic target_* columns.
    """
    result = table.copy()
    result["dataset_name"] = dataset_name

    if "target_type" not in result.columns:
        result["target_type"] = "gauge"
    if "target_name" not in result.columns and "gauge_name" in result.columns:
        result["target_name"] = result["gauge_name"]
    if "target_rain_mm_h" not in result.columns and "gauge_rain_mm_h" in result.columns:
        result["target_rain_mm_h"] = result["gauge_rain_mm_h"]
    if "target_rain_mm" not in result.columns and "gauge_rain_mm" in result.columns:
        result["target_rain_mm"] = result["gauge_rain_mm"]
    if "distance_to_target_km" not in result.columns and "distance_to_gauge_km" in result.columns:
        result["distance_to_target_km"] = result["distance_to_gauge_km"]

    for column in CANONICAL_RAIN_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    ordered_columns = CANONICAL_RAIN_COLUMNS + [
        column for column in result.columns if column not in CANONICAL_RAIN_COLUMNS
    ]
    return result[ordered_columns]


def validate_model_ready_table(table: pd.DataFrame) -> None:
    """Raise a clear error if the prepared table is not usable for modeling."""
    missing = [column for column in REQUIRED_MODEL_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Prepared table is missing required columns: {missing}")

    if table.empty:
        raise ValueError("Prepared table is empty")
    if table["target_rain_mm_h"].isna().all():
        raise ValueError("Prepared table has no target_rain_mm_h labels")
    if table["attenuation"].isna().all():
        raise ValueError("Prepared table has no attenuation values")
    required_missing = [column for column in REQUIRED_MODEL_COLUMNS if table[column].isna().all()]
    if required_missing:
        raise ValueError(f"Prepared table has only missing values for: {required_missing}")


def write_prepared_table(table: pd.DataFrame, output_path: Path) -> None:
    """Validate and write a canonical prepared table."""
    validate_model_ready_table(table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
