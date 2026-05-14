"""
Phase 3 training and evaluation utilities for rain-rate estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from .rain_models import RainMLP, power_law_rain_rate
from ..phase2.rain_schema import validate_model_ready_table


FEATURE_COLUMNS = [
    "attenuation",
    "path_loss",
    "rsl",
    "tsl",
    "link_length_km",
    "frequency_GHz",
    "polarization_vertical",
]


@dataclass
class RainSplit:
    features: torch.Tensor
    target: torch.Tensor
    attenuation: torch.Tensor
    link_length: torch.Tensor
    frame: pd.DataFrame


@dataclass
class RainDataset:
    train: RainSplit
    val: RainSplit
    test: RainSplit
    feature_columns: list[str]
    feature_mean: pd.Series
    feature_std: pd.Series


def clone_with_limited_training(
    dataset: RainDataset,
    train_fraction: float,
    seed: int,
    target_noise_std: float = 0.0,
    feature_missing_prob: float = 0.0,
) -> RainDataset:
    """
    Create a copy with corrupted/limited training data only.

    - train_fraction randomly drops supervised training rows.
    - target_noise_std adds Gaussian label noise relative to train target std.
    - feature_missing_prob masks standardized training features to 0, i.e. train
      mean imputation after standardization.

    Validation and test splits remain clean.
    """
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if target_noise_std < 0.0:
        raise ValueError("target_noise_std must be non-negative")
    if not 0.0 <= feature_missing_prob < 1.0:
        raise ValueError("feature_missing_prob must be in [0, 1)")

    generator = torch.Generator(device=dataset.train.features.device)
    generator.manual_seed(seed)

    n_train = dataset.train.features.shape[0]
    n_keep = max(1, int(n_train * train_fraction))
    indices = torch.randperm(n_train, generator=generator, device=dataset.train.features.device)[:n_keep]
    indices, _ = torch.sort(indices)

    features = dataset.train.features[indices].clone()
    target = dataset.train.target[indices].clone()
    attenuation = dataset.train.attenuation[indices].clone()
    link_length = dataset.train.link_length[indices].clone()
    frame = dataset.train.frame.iloc[indices.detach().cpu().numpy()].reset_index(drop=True)

    if target_noise_std > 0.0:
        scale = dataset.train.target.std().clamp(min=1e-8)
        noise = torch.randn(target.shape, generator=generator, device=target.device) * target_noise_std * scale
        target = torch.clamp(target + noise, min=0.0)

    if feature_missing_prob > 0.0:
        mask = torch.rand(features.shape, generator=generator, device=features.device) < feature_missing_prob
        features = features.masked_fill(mask, 0.0)

    return RainDataset(
        train=RainSplit(features, target, attenuation, link_length, frame),
        val=dataset.val,
        test=dataset.test,
        feature_columns=dataset.feature_columns,
        feature_mean=dataset.feature_mean,
        feature_std=dataset.feature_std,
    )


def load_rain_table(path: Path) -> pd.DataFrame:
    """Load the Phase 2 split table and validate model-required columns."""
    table = pd.read_csv(path)
    table["time_utc"] = pd.to_datetime(table["time_utc"], utc=True)
    validate_model_ready_table(table)
    if "split" not in table.columns:
        raise ValueError("Phase 3 input must include a chronological 'split' column")
    for split_name in ["train", "val", "test"]:
        if (table["split"] == split_name).sum() == 0:
            raise ValueError(f"Input table has no rows for split '{split_name}'")
    return table.sort_values(["time_utc", "sublink_id"]).reset_index(drop=True)


def build_feature_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Create numeric model features from the canonical table."""
    features = table.copy()
    features["polarization_vertical"] = (
        features["polarization"].astype(str).str.lower().str.startswith("v").astype(float)
    )
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    if features[FEATURE_COLUMNS].isna().any().any():
        missing = features[FEATURE_COLUMNS].isna().mean()
        raise ValueError(f"Feature table contains missing values:\n{missing[missing > 0]}")
    return features[FEATURE_COLUMNS]


def prepare_rain_dataset(path: Path, device: str = "cpu") -> RainDataset:
    """Prepare standardized train/validation/test tensors."""
    table = load_rain_table(path)
    feature_frame = build_feature_frame(table)
    train_mask = table["split"] == "train"
    feature_mean = feature_frame.loc[train_mask].mean()
    feature_std = feature_frame.loc[train_mask].std().replace(0.0, 1.0).fillna(1.0)
    standardized = (feature_frame - feature_mean) / feature_std #z-score normalization

    def make_split(split_name: str) -> RainSplit:
        mask = table["split"] == split_name
        frame = table.loc[mask].copy().reset_index(drop=True)
        x = torch.tensor(standardized.loc[mask].to_numpy(), dtype=torch.float32, device=device)
        y = torch.tensor(
            frame["target_rain_mm_h"].to_numpy().reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )
        attenuation = torch.tensor(
            frame["attenuation"].to_numpy().reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )
        link_length = torch.tensor(
            frame["link_length_km"].to_numpy().reshape(-1, 1),
            dtype=torch.float32,
            device=device,
        )
        return RainSplit(x, y, attenuation, link_length, frame)

    return RainDataset(
        train=make_split("train"),
        val=make_split("val"),
        test=make_split("test"),
        feature_columns=FEATURE_COLUMNS,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def rain_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wet_threshold_mm_h: float = 0.1,
) -> dict[str, float]:
    """Regression and wet/dry metrics for rain-rate predictions."""
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(mse))
    wet_true = true > wet_threshold_mm_h
    wet_pred = pred > wet_threshold_mm_h
    rainy_rmse = float(np.sqrt(np.mean((pred[wet_true] - true[wet_true]) ** 2))) if wet_true.any() else float("nan")
    rainy_mae = float(np.mean(np.abs(pred[wet_true] - true[wet_true]))) if wet_true.any() else float("nan")
    dry_mae = float(np.mean(np.abs(pred[~wet_true] - true[~wet_true]))) if (~wet_true).any() else float("nan")
    tp = float(np.logical_and(wet_true, wet_pred).sum())
    fp = float(np.logical_and(~wet_true, wet_pred).sum())
    fn = float(np.logical_and(wet_true, ~wet_pred).sum())
    tn = float(np.logical_and(~wet_true, ~wet_pred).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rainy_rmse": rainy_rmse,
        "rainy_mae": rainy_mae,
        "dry_mae": dry_mae,
        "bias": float(np.mean(pred - true)),
        "wet_precision": float(precision),
        "wet_recall": float(recall),
        "wet_f1": float(f1),
        "wet_accuracy": float(accuracy),
    }


def evaluate_predictions(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wet_threshold_mm_h: float,
) -> dict[str, float | str]:
    """Create a metric row for one model and split."""
    row: dict[str, float | str] = {
        "model": model_name,
        "split": split_name,
    }
    row.update(rain_metrics(y_true, y_pred, wet_threshold_mm_h=wet_threshold_mm_h))
    return row


def evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    dataset: RainDataset,
    wet_threshold_mm_h: float,
) -> tuple[list[dict[str, float | str]], pd.DataFrame]:
    """Evaluate a trained neural model on all splits."""
    model.eval()
    rows: list[dict[str, float | str]] = []
    prediction_frames = []
    with torch.no_grad():
        for split_name in ["train", "val", "test"]:
            split = getattr(dataset, split_name)
            pred = model(split.features).cpu().numpy().reshape(-1)
            true = split.target.cpu().numpy().reshape(-1)
            rows.append(evaluate_predictions(model_name, split_name, true, pred, wet_threshold_mm_h))
            frame = split.frame[["time_utc", "sublink_id", "target_name", "target_rain_mm_h", "split"]].copy()
            frame["model"] = model_name
            frame["prediction_mm_h"] = pred
            prediction_frames.append(frame)
    return rows, pd.concat(prediction_frames, ignore_index=True)


def evaluate_physics_baseline(
    dataset: RainDataset,
    power_a: float,
    power_b: float,
    wet_threshold_mm_h: float,
) -> tuple[list[dict[str, float | str]], pd.DataFrame]:
    """Evaluate the pure power-law baseline on all splits."""
    rows: list[dict[str, float | str]] = []
    prediction_frames = []
    for split_name in ["train", "val", "test"]:
        split = getattr(dataset, split_name)
        pred_tensor = power_law_rain_rate(split.attenuation, split.link_length, power_a, power_b)
        pred = pred_tensor.detach().cpu().numpy().reshape(-1)
        true = split.target.detach().cpu().numpy().reshape(-1)
        rows.append(evaluate_predictions("physics", split_name, true, pred, wet_threshold_mm_h))
        frame = split.frame[["time_utc", "sublink_id", "target_name", "target_rain_mm_h", "split"]].copy()
        frame["model"] = "physics"
        frame["prediction_mm_h"] = pred
        prediction_frames.append(frame)
    return rows, pd.concat(prediction_frames, ignore_index=True)


def calibrate_power_a(dataset: RainDataset, power_b: float) -> float:
    """
    Fit power-law coefficient a on the training split only.

    We keep b fixed and solve least squares for A ~= a * L * R^b. This gives a
    dataset-specific physics coefficient without leaking validation/test labels.
    """
    rain = torch.clamp(dataset.train.target, min=0.0)
    design = dataset.train.link_length * torch.pow(rain + 1e-8, power_b)
    numerator = torch.sum(design * dataset.train.attenuation)
    denominator = torch.sum(design * design) + 1e-12
    return float((numerator / denominator).detach().cpu().item())


def append_physics_prior_feature(
    dataset: RainDataset,
    power_a: float,
    power_b: float,
    standardize_prior: bool = False,
) -> RainDataset:
    """
    Append raw power-law rain estimate as a final feature.

    For residual models this prior should remain raw. For feature-only models it
    can be standardized with training statistics like the other inputs.
    """
    train_prior = power_law_rain_rate(dataset.train.attenuation, dataset.train.link_length, power_a, power_b)
    prior_mean = train_prior.mean()
    prior_std = train_prior.std().clamp(min=1e-8)

    def append(split: RainSplit) -> RainSplit:
        prior = power_law_rain_rate(split.attenuation, split.link_length, power_a, power_b)
        feature_prior = (prior - prior_mean) / prior_std if standardize_prior else prior
        features = torch.cat([split.features, feature_prior], dim=1)
        frame = split.frame.copy()
        frame["physics_prior_mm_h"] = prior.detach().cpu().numpy().reshape(-1)
        return RainSplit(features, split.target, split.attenuation, split.link_length, frame)

    return RainDataset(
        train=append(dataset.train),
        val=append(dataset.val),
        test=append(dataset.test),
        feature_columns=dataset.feature_columns + ["physics_prior_mm_h"],
        feature_mean=dataset.feature_mean,
        feature_std=dataset.feature_std,
    )


def train_model(
    model: RainMLP,
    loss_fn,
    dataset: RainDataset,
    n_epochs: int = 1000,
    lr: float = 1e-3,
) -> list[dict[str, float]]:
    """Train a rain model using full-batch optimization."""
    optimizer = Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss, loss_dict = loss_fn(
            model,
            dataset.train.features,
            dataset.train.target,
            dataset.train.attenuation,
            dataset.train.link_length,
        )
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, n_epochs // 20) == 0:
            history.append({"epoch": epoch + 1, **loss_dict})
    return history
