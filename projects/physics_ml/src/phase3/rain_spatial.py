#!/usr/bin/env python3
"""
Phase 3C spatial CML experiment.

Builds a gauge-level wide table from the long canonical CML table:
one row = one timestamp + one gauge target, with multiple nearby CML links as
features. This directly tests whether adding spatial CML information improves
over a one-link representation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .rain_models import RainMLP
from .rain_train import rain_metrics


LINK_FEATURES = ["attenuation", "path_loss", "rsl", "tsl", "link_length_km", "frequency_GHz", "r_physics"]


@dataclass
class SpatialSplit:
    features: torch.Tensor
    target: torch.Tensor
    frame: pd.DataFrame


@dataclass
class SpatialDataset:
    train: SpatialSplit
    val: SpatialSplit
    test: SpatialSplit
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 3C spatial CML comparison")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "phase2_spatial_validation" / "prepared_with_splits.csv",
        help="Validated long table with split column and multiple CML links per target",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "phase3c_spatial",
        help="Directory for spatial metrics and plots",
    )
    parser.add_argument("--max-links-per-target", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--power-b", type=float, default=0.8)
    parser.add_argument("--wet-threshold-mm-h", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def calibrate_power_a_from_frame(frame: pd.DataFrame, power_b: float) -> float:
    """Fit power-law coefficient a on train rows only."""
    train = frame[frame["split"] == "train"].copy()
    rain = train["target_rain_mm_h"].clip(lower=0).to_numpy()
    design = train["link_length_km"].to_numpy() * np.power(rain + 1e-8, power_b)
    attenuation = train["attenuation"].to_numpy()
    return float(np.sum(design * attenuation) / (np.sum(design * design) + 1e-12))


def build_spatial_wide_table(
    input_path: Path,
    power_b: float,
    max_links_per_target: int,
) -> tuple[pd.DataFrame, float]:
    """Convert long CML rows to one row per target/timestamp with link-ranked features."""
    long = pd.read_csv(input_path)
    long["time_utc"] = pd.to_datetime(long["time_utc"], utc=True)
    numeric_columns = ["target_rain_mm_h", "attenuation", "path_loss", "rsl", "tsl", "link_length_km", "frequency_GHz"]
    for column in numeric_columns:
        long[column] = pd.to_numeric(long[column], errors="coerce")
    long["polarization_vertical"] = long["polarization"].astype(str).str.lower().str.startswith("v").astype(float)
    power_a = calibrate_power_a_from_frame(long, power_b)
    long["r_physics"] = np.power(
        np.clip(long["attenuation"] / (power_a * long["link_length_km"] + 1e-8), 0.0, None),
        1.0 / power_b,
    )

    rank_frames = []
    for target_name, group in long.groupby("target_name"):
        sublinks = rank_sublinks_for_target(group)[:max_links_per_target]
        rank_map = {sublink_id: rank for rank, sublink_id in enumerate(sublinks)}
        target_group = group[group["sublink_id"].isin(sublinks)].copy()
        target_group["link_rank"] = target_group["sublink_id"].map(rank_map)
        rank_frames.append(target_group)
    ranked = pd.concat(rank_frames, ignore_index=True)

    base = (
        ranked[["time_utc", "target_name", "split", "target_rain_mm_h"]]
        .drop_duplicates()
        .sort_values(["target_name", "time_utc"])
        .reset_index(drop=True)
    )
    wide = base.copy()
    for rank in range(max_links_per_target):
        rank_data = ranked[ranked["link_rank"] == rank]
        wide[f"link{rank}_present"] = 0.0
        for feature in LINK_FEATURES + ["polarization_vertical"]:
            pivot = rank_data[["time_utc", "target_name", feature]].rename(columns={feature: f"link{rank}_{feature}"})
            wide = wide.merge(pivot, on=["time_utc", "target_name"], how="left")
        wide.loc[wide[f"link{rank}_attenuation"].notna(), f"link{rank}_present"] = 1.0

    return wide.sort_values(["time_utc", "target_name"]).reset_index(drop=True), power_a


def rank_sublinks_for_target(group: pd.DataFrame) -> list[int]:
    """
    Rank target-linked CMLs using training data only.

    PyNNcml already limits links to nearby gauges. Within that candidate set,
    rank by absolute Spearman rain/attenuation relationship on the training
    split, then by attenuation variance as a deterministic fallback.
    """
    rows = []
    for sublink_id, sublink_group in group.groupby("sublink_id"):
        train = sublink_group[sublink_group["split"] == "train"]
        corr = train[["target_rain_mm_h", "attenuation"]].corr(method="spearman").iloc[0, 1]
        if pd.isna(corr):
            corr = 0.0
        rows.append(
            {
                "sublink_id": int(sublink_id),
                "abs_corr": abs(float(corr)),
                "attenuation_std": float(train["attenuation"].std()) if len(train) else 0.0,
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["abs_corr", "attenuation_std", "sublink_id"],
        ascending=[False, False, True],
    )
    return ranked["sublink_id"].astype(int).tolist()


def prepare_spatial_dataset(
    wide: pd.DataFrame,
    mode: str,
    max_links_per_target: int,
    device: str,
) -> SpatialDataset:
    """Create train/val/test tensors for one-link or spatial feature modes."""
    if mode == "single_link":
        ranks = [0]
        include_physics = False
    elif mode == "single_link_physics":
        ranks = [0]
        include_physics = True
    elif mode == "spatial":
        ranks = list(range(max_links_per_target))
        include_physics = False
    elif mode == "spatial_physics":
        ranks = list(range(max_links_per_target))
        include_physics = True
    else:
        raise ValueError(f"Unknown spatial mode: {mode}")

    feature_columns = []
    base_features = ["attenuation", "path_loss", "rsl", "tsl", "link_length_km", "frequency_GHz", "polarization_vertical"]
    if include_physics:
        base_features.append("r_physics")
    for rank in ranks:
        feature_columns.append(f"link{rank}_present")
        feature_columns.extend([f"link{rank}_{feature}" for feature in base_features])

    features = wide[feature_columns].copy()
    train_mask = wide["split"] == "train"
    train_mean = features.loc[train_mask].mean()
    features = features.fillna(train_mean)
    train_std = features.loc[train_mask].std().replace(0.0, 1.0).fillna(1.0)
    standardized = (features - train_mean) / train_std
    standardized = standardized.fillna(0.0)
 
    def make_split(split: str) -> SpatialSplit:
        mask = wide["split"] == split
        frame = wide.loc[mask, ["time_utc", "target_name", "target_rain_mm_h", "split"]].copy().reset_index(drop=True)
        x = torch.tensor(standardized.loc[mask].to_numpy(), dtype=torch.float32, device=device)
        y = torch.tensor(frame["target_rain_mm_h"].to_numpy().reshape(-1, 1), dtype=torch.float32, device=device)
        return SpatialSplit(x, y, frame)

    return SpatialDataset(
        train=make_split("train"),
        val=make_split("val"),
        test=make_split("test"),
        feature_columns=feature_columns,
    )


def train_spatial_model(
    dataset: SpatialDataset,
    epochs: int,
    lr: float,
    seed: int,
    device: str,
    patience: int = 100,
    hidden_dims: list[int] | None = None,
    rain_weight_alpha: float = 0.0,
    wet_threshold_mm_h: float = 0.1,
) -> RainMLP:
    """Train with validation-based early stopping and best-weight restore."""
    torch.manual_seed(seed)
    model = RainMLP(dataset.train.features.shape[1], hidden_dims=hidden_dims or [64, 64]).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    stale = 0

    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(dataset.train.features)
        if rain_weight_alpha > 0.0:
            weights = 1.0 + rain_weight_alpha * (dataset.train.target > wet_threshold_mm_h).float()
            loss = torch.mean(weights * (prediction - dataset.train.target) ** 2)
        else:
            loss = mse(prediction, dataset.train.target)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(dataset.val.features)
            val_loss = mse(val_prediction, dataset.val.target).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model


def evaluate_spatial_model(
    model_name: str,
    model: RainMLP,
    dataset: SpatialDataset,
    wet_threshold_mm_h: float,
) -> tuple[list[dict[str, float | str]], pd.DataFrame]:
    rows = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            split_data = getattr(dataset, split)
            pred = model(split_data.features).cpu().numpy().reshape(-1)
            true = split_data.target.cpu().numpy().reshape(-1)
            row = {"model": model_name, "split": split}
            row.update(rain_metrics(true, pred, wet_threshold_mm_h=wet_threshold_mm_h))
            rows.append(row)
            frame = split_data.frame.copy()
            frame["model"] = model_name
            frame["prediction_mm_h"] = pred
            predictions.append(frame)
    return rows, pd.concat(predictions, ignore_index=True)


def _dataframe_to_markdown(frame: pd.DataFrame) -> str:
    rendered = frame.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: f"{value:.6g}")
        else:
            rendered[column] = rendered[column].astype(str)
    columns = list(rendered.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def plot_spatial_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    test = metrics[metrics["split"] == "test"].sort_values("rmse")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(
        axes,
        ["rmse", "rainy_rmse", "wet_f1"],
        ["All-sample RMSE", "Rainy-only RMSE", "Wet/Dry F1"],
    ):
        ax.bar(test["model"], test[metric])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_spatial_report(
    output_path: Path,
    input_path: Path,
    metrics: pd.DataFrame,
    wide: pd.DataFrame,
    power_a: float,
    power_b: float,
) -> None:
    test = metrics[metrics["split"] == "test"].sort_values("rmse")
    single_rmse = float(test[test["model"] == "single_link_nn"]["rmse"].iloc[0])
    best = test.iloc[0]
    improvement = (single_rmse - float(best["rmse"])) / single_rmse * 100.0
    link_counts = wide.filter(like="_present").sum(axis=1)
    lines = [
        "# Phase 3C Spatial CML Modeling Report",
        "",
        f"Input: `{input_path}`",
        f"Rows: `{len(wide)}` gauge-time samples",
        f"Targets: `{wide['target_name'].nunique()}`",
        f"Mean available links per row: `{link_counts.mean():.3g}`",
        f"Power law: train-calibrated `a={power_a:.6g}`, fixed `b={power_b:.6g}`",
        "",
        "## Test Metrics",
        "",
        _dataframe_to_markdown(test[["model", "rmse", "rainy_rmse", "mae", "wet_f1", "wet_accuracy"]]),
        "",
        "## Result Summary",
        "",
        f"- Best model: `{best['model']}` with RMSE `{float(best['rmse']):.6g}`.",
        f"- Improvement vs one-link NN: `{improvement:.3g}%`.",
        "- This tests whether adding multiple nearby CML links gives a better local rain-field proxy than one link alone.",
        "",
        "## Visual Artifacts",
        "",
        "- `spatial_metric_bars.png`: all-sample RMSE, rainy-only RMSE, and wet/dry F1.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = min(args.epochs, 300) if args.quick else args.epochs

    wide, power_a = build_spatial_wide_table(args.input, args.power_b, args.max_links_per_target)
    wide.to_csv(output_dir / "spatial_wide_table.csv", index=False)

    modes = ["single_link", "single_link_physics", "spatial", "spatial_physics"]
    metrics_rows = []
    prediction_tables = []
    for mode in modes:
        dataset = prepare_spatial_dataset(wide, mode=mode, max_links_per_target=args.max_links_per_target, device=args.device)
        model = train_spatial_model(
            dataset,
            epochs=epochs,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            wet_threshold_mm_h=args.wet_threshold_mm_h,
        )
        metrics, predictions = evaluate_spatial_model(mode + "_nn", model, dataset, args.wet_threshold_mm_h)
        metrics_rows.extend(metrics)
        prediction_tables.append(predictions)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_tables, ignore_index=True)
    metrics_df.to_csv(output_dir / "spatial_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "spatial_predictions.csv", index=False)
    plot_spatial_metrics(metrics_df, output_dir / "spatial_metric_bars.png")
    write_spatial_report(output_dir / "spatial_report.md", args.input, metrics_df, wide, power_a, args.power_b)

    test = metrics_df[metrics_df["split"] == "test"].sort_values("rmse")
    print("\nSpatial test metrics")
    print("-" * 72)
    print(
        test[["model", "rmse", "rainy_rmse", "mae", "wet_f1", "wet_accuracy"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nSaved spatial report: {output_dir / 'spatial_report.md'}")


if __name__ == "__main__":
    main()
