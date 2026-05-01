"""
OpenMRG rain/CML data preparation for Phase 2.

This module owns data loading, gauge/link selection, time alignment, and basic
attenuation preparation. It intentionally does not contain model or training
code; those belong in rain_models.py and rain_train.py later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .rain_schema import canonicalize_rain_table


DEFAULT_OPENMRG_ROOT = (
    Path(__file__).resolve().parents[3]
    / "dataset"
    / "open_datasets"
    / "OpenMRG_Sweden"
)

OPENMRG_DATASET_NAME = "openmrg_sweden"
MISSING_SIGNAL_VALUE = 1e10


@dataclass(frozen=True)
class OpenMrgPaths:
    """Canonical file paths for the OpenMRG Sweden dataset."""

    root: Path
    cml_nc: Path
    cml_metadata: Path
    city_gauges: Path
    city_gauge_metadata: Path
    smhi_gauge: Path
    smhi_gauge_metadata: Path


def resolve_openmrg_paths(root: Path | str | None = None) -> OpenMrgPaths:
    """Return the expected OpenMRG file layout."""
    dataset_root = Path(root) if root is not None else DEFAULT_OPENMRG_ROOT
    return OpenMrgPaths(
        root=dataset_root,
        cml_nc=dataset_root / "cml" / "cml.nc",
        cml_metadata=dataset_root / "cml" / "cml_metadata.csv",
        city_gauges=dataset_root / "gauges" / "city" / "CityGauges-2015JJA.csv",
        city_gauge_metadata=dataset_root / "gauges" / "city" / "CityGauges-metadata.csv",
        smhi_gauge=dataset_root / "gauges" / "smhi" / "GbgA-71420-2015JJA.csv",
        smhi_gauge_metadata=dataset_root / "gauges" / "smhi" / "GbgA-71420-metadata.csv",
    )


def check_data_availability(paths: OpenMrgPaths) -> dict[str, bool]:
    """Report whether each expected Stage 1 input exists."""
    return {
        "root": paths.root.exists(),
        "cml_nc": paths.cml_nc.exists(),
        "cml_metadata": paths.cml_metadata.exists(),
        "city_gauges": paths.city_gauges.exists(),
        "city_gauge_metadata": paths.city_gauge_metadata.exists(),
        "smhi_gauge": paths.smhi_gauge.exists(),
        "smhi_gauge_metadata": paths.smhi_gauge_metadata.exists(),
    }


def load_city_gauges(paths: OpenMrgPaths) -> pd.DataFrame:
    """Load 1-minute city rain gauge amounts."""
    gauges = pd.read_csv(paths.city_gauges, na_values=["NA", ""])
    gauges["time_utc"] = pd.to_datetime(gauges.pop("Time_UTC"), utc=True)
    return gauges


def load_city_gauge_metadata(paths: OpenMrgPaths) -> pd.DataFrame:
    """Load city rain gauge locations and metadata."""
    return pd.read_csv(paths.city_gauge_metadata)


def load_cml_metadata(paths: OpenMrgPaths) -> pd.DataFrame:
    """Load CML sublink locations and radio metadata."""
    return pd.read_csv(paths.cml_metadata)


def available_city_gauges(paths: OpenMrgPaths) -> list[str]:
    """Return city gauge column names available in the 1-minute gauge file."""
    gauges = load_city_gauges(paths)
    return [col for col in gauges.columns if col != "time_utc"]


def load_city_gauge_rate(paths: OpenMrgPaths, gauge_name: str) -> pd.DataFrame:
    """
    Load a single city gauge and convert 1-minute amount to rain rate.

    The OpenMRG city gauge file stores accumulated rain in millimetres for the
    preceding minute. The CML power law uses rain rate in mm/h, so Stage 1
    normalizes labels to mm/h.
    """
    gauges = load_city_gauges(paths)
    if gauge_name not in gauges.columns:
        available = ", ".join(available_city_gauges(paths))
        raise ValueError(f"Unknown gauge '{gauge_name}'. Available gauges: {available}")

    amount = pd.to_numeric(gauges[gauge_name], errors="coerce")
    return pd.DataFrame(
        {
            "time_utc": gauges["time_utc"],
            "gauge_name": gauge_name,
            "gauge_rain_mm": amount,
            "gauge_rain_mm_h": amount * 60.0,
        }
    )


def _latlon_to_xy_km(
    lat: pd.Series | np.ndarray | float,
    lon: pd.Series | np.ndarray | float,
    ref_lat: float,
    ref_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project local lat/lon coordinates to approximate x/y kilometres."""
    earth_radius_km = 6371.0
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    x = np.deg2rad(lon_arr - ref_lon) * earth_radius_km * np.cos(np.deg2rad(ref_lat))
    y = np.deg2rad(lat_arr - ref_lat) * earth_radius_km
    return x, y


def _point_to_segment_distance_km(
    point_x: float,
    point_y: float,
    start_x: np.ndarray,
    start_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray,
) -> np.ndarray:
    """Distance from a point to each CML path segment in kilometres."""
    seg_x = end_x - start_x
    seg_y = end_y - start_y
    seg_len_sq = seg_x**2 + seg_y**2
    seg_len_sq = np.where(seg_len_sq == 0, np.nan, seg_len_sq)

    t = ((point_x - start_x) * seg_x + (point_y - start_y) * seg_y) / seg_len_sq
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    nearest_x = start_x + t * seg_x
    nearest_y = start_y + t * seg_y
    return np.sqrt((point_x - nearest_x) ** 2 + (point_y - nearest_y) ** 2)


def find_nearest_links_to_gauge(
    gauge_name: str,
    gauge_metadata: pd.DataFrame,
    cml_metadata: pd.DataFrame,
    max_links: int = 5,
) -> pd.DataFrame:
    """
    Rank CML sublinks by distance from gauge point to CML path.

    The CML measures path-averaged attenuation, so distance to the line segment
    between endpoints is more relevant than distance to either endpoint alone.
    """
    gauge_row = gauge_metadata[
        gauge_metadata["Name"].astype(str).str.lower() == gauge_name.lower()
    ]
    if gauge_row.empty:
        available = ", ".join(gauge_metadata["Name"].astype(str).tolist())
        raise ValueError(f"Unknown gauge '{gauge_name}'. Available gauges: {available}")

    gauge = gauge_row.iloc[0]
    gauge_lat = float(gauge["Latitude_DecDeg"])
    gauge_lon = float(gauge["Longitude_DecDeg"])

    start_x, start_y = _latlon_to_xy_km(
        cml_metadata["NearLatitude_DecDeg"],
        cml_metadata["NearLongitude_DecDeg"],
        gauge_lat,
        gauge_lon,
    )
    end_x, end_y = _latlon_to_xy_km(
        cml_metadata["FarLatitude_DecDeg"],
        cml_metadata["FarLongitude_DecDeg"],
        gauge_lat,
        gauge_lon,
    )
    point_x, point_y = 0.0, 0.0

    ranked = cml_metadata.copy()
    ranked["gauge_name"] = gauge_name
    ranked["gauge_latitude"] = gauge_lat
    ranked["gauge_longitude"] = gauge_lon
    ranked["distance_to_gauge_km"] = _point_to_segment_distance_km(
        point_x, point_y, start_x, start_y, end_x, end_y
    )
    return ranked.sort_values("distance_to_gauge_km").head(max_links).reset_index(drop=True)


def _import_netcdf4():
    try:
        import netCDF4  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "netCDF4 is required to read OpenMRG cml.nc. "
            "Install project requirements or run: pip install netCDF4"
        ) from exc
    return netCDF4


def _netcdf_time_to_utc(time_var) -> pd.DatetimeIndex:
    """Convert NetCDF time values to a UTC pandas DatetimeIndex."""
    netcdf4 = _import_netcdf4()
    values = np.asarray(time_var[:])
    units = getattr(time_var, "units", None)
    if units:
        try:
            datetimes = netcdf4.num2date(
                values,
                units,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            return pd.to_datetime(datetimes, utc=True)
        except Exception:
            pass
    return pd.to_datetime(values, unit="s", utc=True)


def _read_netcdf_variable_for_sublink(nc, variable_name: str, sublink_index: int) -> np.ndarray:
    """Read a time series for one sublink from a 2D NetCDF variable."""
    variable = nc.variables[variable_name]
    dimensions = tuple(getattr(variable, "dimensions", ()))

    if len(variable.shape) != 2:
        raise ValueError(f"Expected '{variable_name}' to be 2D, got shape {variable.shape}")

    if dimensions and dimensions[0].lower() == "time":
        values = variable[:, sublink_index]
    elif dimensions and dimensions[-1].lower() == "time":
        values = variable[sublink_index, :]
    else:
        values = variable[:, sublink_index]

    array = np.ma.filled(values, np.nan).astype(float)
    array[array == MISSING_SIGNAL_VALUE] = np.nan
    return array


def load_cml_signal(
    paths: OpenMrgPaths,
    sublink_id: int,
    variables: Iterable[str] = ("rsl", "tsl"),
) -> pd.DataFrame:
    """Load RSL/TSL time series for one CML sublink."""
    if not paths.cml_nc.exists():
        raise FileNotFoundError(
            f"Missing CML signal file: {paths.cml_nc}. "
            "Download or restore OpenMRG cml.nc before building the Stage 1 table."
        )

    netcdf4 = _import_netcdf4()
    with netcdf4.Dataset(paths.cml_nc, "r") as nc:
        if "sublink" not in nc.variables:
            raise KeyError("Expected NetCDF variable 'sublink' in OpenMRG cml.nc")

        sublinks = np.asarray(nc.variables["sublink"][:]).astype(int)
        matches = np.where(sublinks == int(sublink_id))[0]
        if len(matches) == 0:
            raise ValueError(f"Sublink {sublink_id} not found in cml.nc")
        sublink_index = int(matches[0])

        data = {"time_utc": _netcdf_time_to_utc(nc.variables["time"])}
        for variable_name in variables:
            if variable_name in nc.variables:
                data[variable_name] = _read_netcdf_variable_for_sublink(
                    nc, variable_name, sublink_index
                )

    frame = pd.DataFrame(data).sort_values("time_utc")
    frame["sublink_id"] = int(sublink_id)
    return frame


def aggregate_cml_to_1min(cml_signal: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CML samples to 1-minute means for gauge alignment."""
    if "time_utc" not in cml_signal.columns:
        raise ValueError("CML signal dataframe must include 'time_utc'")

    numeric_columns = [
        col for col in cml_signal.columns if col != "time_utc" and pd.api.types.is_numeric_dtype(cml_signal[col])
    ]
    aggregated = (
        cml_signal.set_index("time_utc")[numeric_columns]
        .resample("1min")
        .mean()
        .reset_index()
    )
    if "sublink_id" in aggregated.columns:
        aggregated["sublink_id"] = aggregated["sublink_id"].round().astype("Int64")
    return aggregated


def compute_attenuation(
    frame: pd.DataFrame,
    method: str = "rsl",
    baseline_quantile: float = 0.95,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, float]:
    """
    Add a simple rain-induced attenuation estimate.

    For Stage 1 we use a dry baseline from the early training portion to avoid
    peeking at validation/test data. If TSL later proves variable, use
    method="path_loss" to compute attenuation from TSL-RSL instead of RSL alone.
    """
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if not 0.0 < baseline_quantile <= 1.0:
        raise ValueError("baseline_quantile must be in (0, 1]")

    result = frame.copy()
    if "time_utc" in result.columns:
        sort_columns = ["time_utc"]
        if "sublink_id" in result.columns:
            sort_columns.append("sublink_id")
        result = result.sort_values(sort_columns).reset_index(drop=True)
    n_train = max(1, int(len(result) * train_fraction))

    if method == "rsl":
        if "rsl" not in result.columns:
            raise ValueError("RSL attenuation requires an 'rsl' column")
        train_part = result.iloc[:n_train]
        baseline = float(train_part["rsl"].quantile(baseline_quantile))
        result["attenuation"] = (baseline - result["rsl"]).clip(lower=0.0)
    elif method == "path_loss":
        if "rsl" not in result.columns or "tsl" not in result.columns:
            raise ValueError("Path-loss attenuation requires 'rsl' and 'tsl' columns")
        result["path_loss"] = result["tsl"] - result["rsl"]
        train_part = result.iloc[:n_train]
        baseline = float(train_part["path_loss"].quantile(1.0 - baseline_quantile))
        result["attenuation"] = (result["path_loss"] - baseline).clip(lower=0.0)
    else:
        raise ValueError("method must be either 'rsl' or 'path_loss'")

    return result, baseline


def build_stage1_table(
    root: Path | str | None = None,
    gauge_name: str = "Chalm",
    sublink_id: int | None = None,
    max_links: int = 1,
    attenuation_method: str = "rsl",
) -> pd.DataFrame:
    """
    Build the first clean OpenMRG table for one gauge and one or more CML links.

    Stage 1 should call this with max_links=1. Later stages can raise max_links
    or pass a specific sublink_id without changing the downstream schema.
    """
    paths = resolve_openmrg_paths(root)
    gauge_metadata = load_city_gauge_metadata(paths)
    cml_metadata = load_cml_metadata(paths)
    gauge_rate = load_city_gauge_rate(paths, gauge_name)

    if sublink_id is not None:
        selected = cml_metadata[cml_metadata["Sublink"].astype(int) == int(sublink_id)].copy()
        if selected.empty:
            raise ValueError(f"Sublink {sublink_id} not found in CML metadata")
    else:
        selected = find_nearest_links_to_gauge(
            gauge_name, gauge_metadata, cml_metadata, max_links=max_links
        )

    tables: list[pd.DataFrame] = []
    for _, link in selected.iterrows():
        signal = load_cml_signal(paths, int(link["Sublink"]))
        cml_1min = aggregate_cml_to_1min(signal)
        aligned = gauge_rate.merge(cml_1min, on="time_utc", how="inner")
        aligned, baseline = compute_attenuation(aligned, method=attenuation_method)

        aligned["link_id"] = int(link["Link"])
        aligned["direction"] = str(link["Direction"])
        aligned["sublink_id"] = int(link["Sublink"])
        aligned["link_length_km"] = float(link["Length_km"])
        aligned["frequency_GHz"] = float(link["Frequency_GHz"])
        aligned["polarization"] = str(link["Polarization"])
        aligned["distance_to_gauge_km"] = float(link.get("distance_to_gauge_km", np.nan))
        aligned["attenuation_method"] = attenuation_method
        aligned["attenuation_baseline"] = baseline
        tables.append(aligned)

    if not tables:
        raise ValueError("No CML links selected for Stage 1 table")

    table = pd.concat(tables, ignore_index=True).sort_values(["time_utc", "sublink_id"])
    return canonicalize_rain_table(table, dataset_name=OPENMRG_DATASET_NAME)


def describe_stage1_table(table: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize a cleaned Stage 1 table for inspection output."""
    summary: dict[str, float | int | str] = {
        "rows": int(len(table)),
        "start_time_utc": str(table["time_utc"].min()),
        "end_time_utc": str(table["time_utc"].max()),
        "sublinks": int(table["sublink_id"].nunique()),
        "rainy_rows": int((table["target_rain_mm_h"] > 0).sum()),
        "rainy_fraction": float((table["target_rain_mm_h"] > 0).mean()),
        "target_rain_missing_fraction": float(table["target_rain_mm_h"].isna().mean()),
        "attenuation_missing_fraction": float(table["attenuation"].isna().mean()),
        "attenuation_mean": float(table["attenuation"].mean()),
        "attenuation_max": float(table["attenuation"].max()),
    }
    if "tsl" in table.columns:
        summary["tsl_std"] = float(table["tsl"].std())
        summary["tsl_range"] = float(table["tsl"].max() - table["tsl"].min())
    return summary
