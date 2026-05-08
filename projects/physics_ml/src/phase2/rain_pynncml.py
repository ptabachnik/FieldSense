"""
PyNNcml adapter for canonical rain/CML preparation.

PyNNcml can download and process OpenMRG into LinkSet objects that already pair
CML links with nearby rain gauges. This adapter converts those objects to the
same table schema used by the rest of Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .rain_schema import canonicalize_rain_table


PYNNCCML_DATASET_NAME = "pynncml_openmrg"


def _require_pynncml():
    try:
        import pynncml as pnc  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pynncml is required for --dataset pynncml_openmrg. "
            "Install it with: pip install pynncml"
        ) from exc
    return pnc


def _time_to_utc(seconds: np.ndarray) -> pd.DatetimeIndex:
    """Convert PyNNcml integer epoch seconds to UTC timestamps."""
    return pd.to_datetime(np.asarray(seconds).astype("int64"), unit="s", utc=True)


def _safe_gauge_name(link_index: int, gauge_index: int) -> str:
    return f"pynncml_link_{link_index}_gauge_{gauge_index}"


def _link_to_table(
    link,
    link_index: int,
    max_label_size: int,
    baseline_train_fraction: float = 0.7,
) -> pd.DataFrame:
    """
    Convert one PyNNcml Link to canonical rows.

    PyNNcml's data_alignment returns signal samples grouped by the rain-gauge
    label time base. We keep Stage 1 simple by averaging RSL/TSL within each
    aligned label interval.
    """
    labels, label_times = link.generate_reference_matrix(max_label_size)
    aligned_labels, rsl_blocks, tsl_blocks, _ = link.data_alignment(max_label_size)

    n = min(len(label_times), aligned_labels.shape[0], rsl_blocks.shape[0], tsl_blocks.shape[0])
    if n == 0:
        return pd.DataFrame()

    labels = aligned_labels[:n]
    rsl = np.nanmean(np.asarray(rsl_blocks[:n], dtype=float), axis=1)
    tsl = np.nanmean(np.asarray(tsl_blocks[:n], dtype=float), axis=1)
    path_loss = tsl - rsl
    n_train = max(1, int(n * baseline_train_fraction))
    baseline = float(np.nanquantile(path_loss[:n_train], 0.05))
    attenuation_1d = np.clip(path_loss - baseline, a_min=0.0, a_max=None)

    metadata = link.meta_data
    gauge_refs = link.gauge_ref if isinstance(link.gauge_ref, list) else [link.gauge_ref]
    gauge_count = labels.shape[1] if labels.ndim == 2 else 1

    rows = []
    for gauge_index in range(gauge_count):
        gauge_ref = gauge_refs[gauge_index] if gauge_index < len(gauge_refs) else None
        gauge_name = _safe_gauge_name(link_index, gauge_index)
        if gauge_ref is not None and hasattr(gauge_ref, "lon") and hasattr(gauge_ref, "lat"):
            gauge_name = f"gauge_{float(gauge_ref.lon):.5f}_{float(gauge_ref.lat):.5f}"

        target = labels[:n, gauge_index] if labels.ndim == 2 else labels[:n]
        frame = pd.DataFrame(
            {
                "time_utc": _time_to_utc(label_times[:n]),
                "target_type": "gauge",
                "target_name": gauge_name,
                "target_rain_mm_h": target,
                "link_id": f"pynncml_link_{link_index}",
                "sublink_id": link_index,
                "direction": pd.NA,
                "rsl": rsl,
                "tsl": tsl,
                "path_loss": path_loss,
                "attenuation": attenuation_1d,
                "attenuation_method": "path_loss_low_quantile",
                "attenuation_baseline": baseline,
                "link_length_km": float(metadata.length),
                "frequency_GHz": float(metadata.frequency),
                "polarization": "Vertical" if bool(metadata.polarization) else "Horizontal",
                "distance_to_target_km": pd.NA,
                "gauge_name": gauge_name,
                "gauge_rain_mm_h": target,
                "gauge_rain_mm": pd.NA,
            }
        )
        rows.append(frame)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_pynncml_openmrg_table(
    data_root: Path | str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    max_links: int = 1,
    link2gauge_distance_m: int = 2000,
    rain_gauge_time_base_s: int = 900,
    window_size_in_min: int = 15,
    baseline_train_fraction: float = 0.7,
) -> pd.DataFrame:
    """
    Download/load PyNNcml OpenMRG and return a canonical prepared table.

    This path is useful when the repo copy of OpenMRG is missing cml.nc. It uses
    PyNNcml's own dataset loader and then converts the resulting LinkSet to our
    Phase 3 schema. PyNNcml's OpenMRG pipeline is most stable at its native
    15-minute rain-gauge time base. The dry attenuation baseline is computed
    from the initial training fraction only to avoid validation/test leakage.
    """
    if not 0.0 < baseline_train_fraction <= 1.0:
        raise ValueError("baseline_train_fraction must be in (0, 1]")

    pnc = _require_pynncml()
    data_path = str(data_root or (Path("data") / "pynncml_openmrg"))
    if not data_path.endswith("/"):
        data_path += "/"

    time_slice = None
    if time_start or time_end:
        time_slice = slice(time_start, time_end)

    link_set, _, _ = pnc.datasets.load_open_mrg(
        data_path=data_path,
        change2min_max=False,
        time_slice=time_slice,
        rain_gauge_time_base=rain_gauge_time_base_s,
        link2gauge_distance=link2gauge_distance_m,
        window_size_in_min=window_size_in_min,
        link_selection=pnc.datasets.xarray_processing.LinkSelection.GAUGEONLY,
    )

    n_links = min(max_links, link_set.n_links)
    tables = [
        _link_to_table(
            link_set.get_link(index),
            index,
            link_set.max_label_size,
            baseline_train_fraction=baseline_train_fraction,
        )
        for index in range(n_links)
    ]
    tables = [table for table in tables if not table.empty]
    if not tables:
        raise ValueError("PyNNcml OpenMRG produced no usable gauge-linked CML rows")

    return canonicalize_rain_table(pd.concat(tables, ignore_index=True), PYNNCCML_DATASET_NAME)
