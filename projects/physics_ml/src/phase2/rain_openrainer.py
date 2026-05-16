"""
OpenRainER Italy adapter for Phase 2 rain/CML preparation.

The dataset ships monthly gzipped NetCDF files. This adapter reads AWS rain
gauges and CML signal levels, selects one gauge plus nearby CML sublinks, and
emits the shared canonical table consumed by Phase 3.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .rain_schema import canonicalize_rain_table


OPENRAINER_DATASET_NAME = "openrainer_italy"
DEFAULT_OPENRAINER_ROOT = (
    Path(__file__).resolve().parents[4]
    / "dataset"
    / "open_datasets"
    / "OpenRainER_Italy"
)


@dataclass(frozen=True)
class OpenRainerPaths:
    root: Path
    aws_dir: Path
    cml_dir: Path


def resolve_openrainer_paths(root: Path | str | None = None) -> OpenRainerPaths:
    dataset_root = Path(root) if root is not None else DEFAULT_OPENRAINER_ROOT
    extracted = dataset_root / "extracted"
    return OpenRainerPaths(
        root=dataset_root,
        aws_dir=extracted / "AWS",
        cml_dir=extracted / "CML",
    )


def _require_netcdf4():
    try:
        import netCDF4  # type: ignore
    except ImportError as exc:
        raise ImportError("netCDF4 is required to read OpenRainER NetCDF files") from exc
    return netCDF4


class _GzipNetcdf:
    """Temporarily decompress a gzipped NetCDF file for netCDF4."""

    def __init__(self, path: Path):
        self.path = path
        self.tmp_path: Path | None = None
        self.dataset = None

    def __enter__(self):
        netcdf4 = _require_netcdf4()
        tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        tmp.close()
        self.tmp_path = Path(tmp.name)
        with gzip.open(self.path, "rb") as src, open(self.tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        self.dataset = netcdf4.Dataset(self.tmp_path)
        return self.dataset

    def __exit__(self, exc_type, exc, tb):
        if self.dataset is not None:
            self.dataset.close()
        if self.tmp_path is not None:
            self.tmp_path.unlink(missing_ok=True)


def _month_starts(time_start: str | None, time_end: str | None) -> list[pd.Timestamp]:
    start = pd.Timestamp(time_start or "2021-01-01", tz="UTC")
    end = pd.Timestamp(time_end or (start + pd.DateOffset(months=1)), tz="UTC")
    if end <= start:
        raise ValueError("time_end must be after time_start")
    months = pd.date_range(start.normalize().replace(day=1), end, freq="MS", tz="UTC")
    if len(months) == 0 or months[0] > start:
        months = months.insert(0, start.normalize().replace(day=1))
    return list(months)


def _aws_file(paths: OpenRainerPaths, month: pd.Timestamp) -> Path:
    return paths.aws_dir / f"AWS_{month:%Y%m}.nc.gz"


def _cml_file(paths: OpenRainerPaths, month: pd.Timestamp) -> Path:
    end = month + pd.offsets.MonthEnd(0)
    return paths.cml_dir / f"CML_{month:%Y%m}010000_{end:%Y%m%d}2359.nc.gz"


def _decode(values) -> list[str]:
    arr = np.asarray(values)
    return [str(value.decode() if isinstance(value, bytes) else value) for value in arr]


def _time_to_utc(seconds: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(seconds).astype("int64"), unit="s", utc=True)


def _latlon_to_xy_km(lat, lon, ref_lat: float, ref_lon: float) -> tuple[np.ndarray, np.ndarray]:
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
    seg_x = end_x - start_x
    seg_y = end_y - start_y
    seg_len_sq = np.where(seg_x**2 + seg_y**2 == 0, np.nan, seg_x**2 + seg_y**2)
    t = ((point_x - start_x) * seg_x + (point_y - start_y) * seg_y) / seg_len_sq
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    nearest_x = start_x + t * seg_x
    nearest_y = start_y + t * seg_y
    return np.sqrt((point_x - nearest_x) ** 2 + (point_y - nearest_y) ** 2)


def _load_aws_month(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with _GzipNetcdf(path) as ds:
        station_ids = _decode(ds.variables["id"][:])
        times = _time_to_utc(ds.variables["time"][:])
        rain = np.ma.filled(ds.variables["rainfall_amount"][:], np.nan).astype(float)
        metadata = pd.DataFrame(
            {
                "target_name": station_ids,
                "latitude": np.ma.filled(ds.variables["latitude"][:], np.nan).astype(float),
                "longitude": np.ma.filled(ds.variables["longitude"][:], np.nan).astype(float),
                "elevation": np.ma.filled(ds.variables["elevation"][:], np.nan).astype(float),
            }
        )

    rain_frame = pd.DataFrame(rain.T, index=times, columns=station_ids)
    return rain_frame, metadata


def _load_cml_metadata(path: Path) -> pd.DataFrame:
    with _GzipNetcdf(path) as ds:
        cml_ids = _decode(ds.variables["cml_id"][:])
        sublink_names = _decode(ds.variables["sublink_id"][:])
        rows = []
        for cml_idx, cml_id in enumerate(cml_ids):
            for sub_idx, sublink_name in enumerate(sublink_names):
                rows.append(
                    {
                        "cml_index": cml_idx,
                        "sublink_index": sub_idx,
                        "cml_id": cml_id,
                        "sublink_name": sublink_name,
                        "sublink_id": cml_idx * len(sublink_names) + sub_idx,
                        "link_length_km": float(ds.variables["length"][cml_idx]) / 1000.0,
                        "site_0_lat": float(ds.variables["site_0_lat"][cml_idx]),
                        "site_0_lon": float(ds.variables["site_0_lon"][cml_idx]),
                        "site_1_lat": float(ds.variables["site_1_lat"][cml_idx]),
                        "site_1_lon": float(ds.variables["site_1_lon"][cml_idx]),
                        "frequency_GHz": float(ds.variables["frequency"][cml_idx, sub_idx]) / 1000.0,
                        "polarization": str(ds.variables["polarization"][cml_idx, sub_idx]),
                    }
                )
    return pd.DataFrame(rows)


def _select_gauge(
    rain_frame: pd.DataFrame,
    aws_metadata: pd.DataFrame,
    cml_metadata: pd.DataFrame,
    gauge_name: str | None,
    max_distance_km: float,
) -> pd.Series:
    if gauge_name and gauge_name.lower() not in {"auto", "chalm"}:
        matches = aws_metadata[aws_metadata["target_name"].astype(str) == gauge_name]
        if matches.empty:
            raise ValueError(f"Unknown OpenRainER gauge '{gauge_name}'")
        return matches.iloc[0]

    rain_totals = rain_frame.sum(axis=0, skipna=True).rename("rain_total_mm").reset_index()
    rain_totals = rain_totals.rename(columns={"index": "target_name"})
    candidates = aws_metadata.merge(rain_totals, on="target_name", how="left")
    candidates = candidates[candidates["rain_total_mm"] > 0].copy()
    if candidates.empty:
        candidates = aws_metadata.copy()
        candidates["rain_total_mm"] = 0.0

    distances = []
    for _, row in candidates.iterrows():
        sx, sy = _latlon_to_xy_km(cml_metadata["site_0_lat"], cml_metadata["site_0_lon"], row["latitude"], row["longitude"])
        ex, ey = _latlon_to_xy_km(cml_metadata["site_1_lat"], cml_metadata["site_1_lon"], row["latitude"], row["longitude"])
        nearest = float(_point_to_segment_distance_km(0.0, 0.0, sx, sy, ex, ey).min())
        distances.append(nearest)
    candidates["nearest_cml_km"] = distances

    nearby = candidates[candidates["nearest_cml_km"] <= max_distance_km]
    if nearby.empty:
        nearby = candidates
    return nearby.sort_values(["rain_total_mm", "nearest_cml_km"], ascending=[False, True]).iloc[0]


def _rank_sublinks_for_gauge(gauge: pd.Series, cml_metadata: pd.DataFrame, max_links: int) -> pd.DataFrame:
    sx, sy = _latlon_to_xy_km(cml_metadata["site_0_lat"], cml_metadata["site_0_lon"], gauge["latitude"], gauge["longitude"])
    ex, ey = _latlon_to_xy_km(cml_metadata["site_1_lat"], cml_metadata["site_1_lon"], gauge["latitude"], gauge["longitude"])
    ranked = cml_metadata.copy()
    ranked["distance_to_target_km"] = _point_to_segment_distance_km(0.0, 0.0, sx, sy, ex, ey)
    return ranked.sort_values(["distance_to_target_km", "sublink_id"]).head(max_links).reset_index(drop=True)


def _load_cml_timeseries(path: Path, selected: pd.DataFrame) -> dict[int, pd.DataFrame]:
    with _GzipNetcdf(path) as ds:
        times = _time_to_utc(ds.variables["time"][:])
        result = {}
        for _, link in selected.iterrows():
            cml_idx = int(link["cml_index"])
            sub_idx = int(link["sublink_index"])
            frame = pd.DataFrame(
                {
                    "time_utc": times,
                    "rsl": np.ma.filled(ds.variables["rsl"][cml_idx, sub_idx, :], np.nan).astype(float),
                    "tsl": np.ma.filled(ds.variables["tsl"][cml_idx, sub_idx, :], np.nan).astype(float),
                }
            )
            result[int(link["sublink_id"])] = frame
    return result


def build_openrainer_table(
    data_root: Path | str | None = None,
    gauge_name: str | None = None,
    max_links: int = 4,
    link2gauge_distance_m: int = 5000,
    time_start: str | None = None,
    time_end: str | None = None,
    baseline_train_fraction: float = 0.7,
) -> pd.DataFrame:
    """Build a canonical OpenRainER rain/CML table for a selected gauge."""
    if not 0.0 < baseline_train_fraction <= 1.0:
        raise ValueError("baseline_train_fraction must be in (0, 1]")
    paths = resolve_openrainer_paths(data_root)
    months = _month_starts(time_start, time_end)
    aws_files = [_aws_file(paths, month) for month in months if _aws_file(paths, month).exists()]
    cml_files = [_cml_file(paths, month) for month in months if _cml_file(paths, month).exists()]
    if not aws_files or not cml_files:
        raise FileNotFoundError("Missing OpenRainER extracted AWS/CML monthly files")

    aws_parts = []
    aws_metadata = None
    for aws_path in aws_files:
        rain_frame, metadata = _load_aws_month(aws_path)
        aws_parts.append(rain_frame)
        aws_metadata = metadata if aws_metadata is None else aws_metadata
    rain_frame = pd.concat(aws_parts).sort_index()

    cml_metadata = _load_cml_metadata(cml_files[0])
    gauge = _select_gauge(
        rain_frame,
        aws_metadata if aws_metadata is not None else pd.DataFrame(),
        cml_metadata,
        gauge_name,
        max_distance_km=link2gauge_distance_m / 1000.0,
    )
    selected = _rank_sublinks_for_gauge(gauge, cml_metadata, max_links=max_links)

    target = rain_frame[str(gauge["target_name"])].rename("target_rain_mm").dropna()
    target_frame = target.reset_index().rename(columns={"index": "time_utc"})
    target_frame["target_rain_mm_h"] = target_frame["target_rain_mm"] * 4.0
    if time_start:
        target_frame = target_frame[target_frame["time_utc"] >= pd.Timestamp(time_start, tz="UTC")]
    if time_end:
        target_frame = target_frame[target_frame["time_utc"] < pd.Timestamp(time_end, tz="UTC")]

    tables = []
    for cml_path in cml_files:
        cml_series = _load_cml_timeseries(cml_path, selected)
        for _, link in selected.iterrows():
            sublink_id = int(link["sublink_id"])
            cml = cml_series[sublink_id].set_index("time_utc").resample("15min").mean().reset_index()
            aligned = target_frame.merge(cml, on="time_utc", how="inner")
            aligned["path_loss"] = aligned["tsl"] - aligned["rsl"]
            aligned = aligned.dropna(subset=["target_rain_mm", "target_rain_mm_h", "rsl", "tsl", "path_loss"])
            if aligned.empty:
                continue
            n_train = max(1, int(len(aligned) * baseline_train_fraction))
            baseline = float(aligned.iloc[:n_train]["path_loss"].quantile(0.05))
            aligned["attenuation"] = (aligned["path_loss"] - baseline).clip(lower=0.0)
            aligned["attenuation_method"] = "path_loss_low_quantile"
            aligned["attenuation_baseline"] = baseline
            aligned["target_type"] = "gauge"
            aligned["target_name"] = str(gauge["target_name"])
            aligned["gauge_name"] = str(gauge["target_name"])
            aligned["gauge_rain_mm"] = aligned["target_rain_mm"]
            aligned["gauge_rain_mm_h"] = aligned["target_rain_mm_h"]
            aligned["link_id"] = str(link["cml_id"])
            aligned["sublink_id"] = sublink_id
            aligned["direction"] = str(link["sublink_name"])
            aligned["link_length_km"] = float(link["link_length_km"])
            aligned["frequency_GHz"] = float(link["frequency_GHz"])
            aligned["polarization"] = str(link["polarization"])
            aligned["distance_to_target_km"] = float(link["distance_to_target_km"])
            tables.append(aligned)

    if not tables:
        raise ValueError("OpenRainER produced no aligned CML/AWS rows")
    table = pd.concat(tables, ignore_index=True).sort_values(["time_utc", "sublink_id"])
    return canonicalize_rain_table(table, OPENRAINER_DATASET_NAME)
