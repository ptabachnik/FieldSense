"""
Rain-rate models for Phase 3.

These models consume the canonical Phase 2 table and predict gauge rain rate
from CML-derived features. The PINN variant uses an algebraic power-law
constraint rather than an ODE.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RainMLP(nn.Module):
    """Small non-negative MLP for interval rain-rate estimation."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        hidden_dims = hidden_dims or [32, 32]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.output = nn.Softplus()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict non-negative rain rate in mm/h."""
        return self.output(self.net(features))


class PhysicsGuidedResidualMLP(nn.Module):
    """
    Physics-guided rain model.

    The last feature is a raw power-law rain estimate R_physics. The network
    learns a correction around that prior instead of learning rain from scratch.
    """

    def __init__(
        self,
        input_dim: int,
        physics_prior_index: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 32]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.correction_net = nn.Sequential(*layers)
        self.physics_prior_index = physics_prior_index
        self.output = nn.Softplus()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict non-negative rain rate as physics prior plus learned correction."""
        physics_prior = torch.clamp(
            features[:, self.physics_prior_index : self.physics_prior_index + 1],
            min=0.0,
        )
        correction = self.correction_net(features)
        return self.output(physics_prior + correction)


def power_law_attenuation(
    rain_rate_mm_h: torch.Tensor,
    link_length_km: torch.Tensor,
    power_a: float,
    power_b: float,
) -> torch.Tensor:
    """Compute expected rain-induced attenuation from power-law physics."""
    rain_rate = torch.clamp(rain_rate_mm_h, min=0.0)
    return power_a * link_length_km * torch.pow(rain_rate + 1e-8, power_b)


def power_law_rain_rate(
    attenuation_db: torch.Tensor,
    link_length_km: torch.Tensor,
    power_a: float,
    power_b: float,
) -> torch.Tensor:
    """Invert the power law to estimate rain rate from attenuation."""
    specific = torch.clamp(attenuation_db, min=0.0) / (power_a * link_length_km + 1e-8)
    return torch.pow(torch.clamp(specific, min=0.0), 1.0 / power_b)


class RainDataLoss:
    """Pure supervised MSE loss."""

    def __init__(self):
        self.mse = nn.MSELoss()

    def __call__(
        self,
        model: nn.Module,
        features: torch.Tensor,
        target: torch.Tensor,
        attenuation: torch.Tensor,
        link_length: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prediction = model(features)
        loss = self.mse(prediction, target)
        return loss, {"total": float(loss.item()), "data": float(loss.item()), "physics": 0.0}


class RainPINNLoss:
    """Supervised rain loss plus power-law attenuation residual."""

    def __init__(self, power_a: float, power_b: float, lambda_physics: float = 0.1):
        self.power_a = power_a
        self.power_b = power_b
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()

    def __call__(
        self,
        model: nn.Module,
        features: torch.Tensor,
        target: torch.Tensor,
        attenuation: torch.Tensor,
        link_length: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prediction = model(features)
        loss_data = self.mse(prediction, target)
        expected_attenuation = power_law_attenuation(
            prediction,
            link_length,
            power_a=self.power_a,
            power_b=self.power_b,
        )
        loss_physics = self.mse(expected_attenuation, attenuation)
        total = loss_data + self.lambda_physics * loss_physics
        return total, {
            "total": float(total.item()),
            "data": float(loss_data.item()),
            "physics": float(loss_physics.item()),
        }
