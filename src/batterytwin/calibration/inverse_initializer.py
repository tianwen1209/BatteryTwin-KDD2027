"""Optional learned warm start for the physics-based calibration optimizer.

The initializer predicts a distribution over Stage-1 calibration parameters from
a measured discharge cycle.  It does not replace PyBaMM optimization: use its
prediction to seed the optimizer's initial population.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .feature_preprocess import build_feature_matrix


SCALAR_KEYS = ["t_cutoff", "V0", "Vend", "Tmax", "ambient_T"]


class InverseInitializer(nn.Module):
    """1D CNN encoder that predicts normalized parameters and uncertainty."""

    def __init__(
        self,
        n_curve_channels: int = 3,
        n_curve_points: int = 256,
        n_scalars: int = 5,
        n_params: int = 11,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_params = n_params
        self.curve_encoder = nn.Sequential(
            nn.Conv1d(n_curve_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.scalar_branch = nn.Sequential(nn.Linear(n_scalars, 32), nn.ReLU())
        self.bottleneck = nn.Sequential(
            nn.Linear(128 + 32, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.mu_head = nn.Linear(hidden_dim, n_params)
        self.log_sigma_head = nn.Linear(hidden_dim, n_params)
        nn.init.constant_(self.mu_head.bias, 0.5)
        nn.init.constant_(self.log_sigma_head.bias, -2.3)

    def forward(
        self, curves: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized parameter means and log standard deviations."""
        curve_features = self.curve_encoder(curves).squeeze(-1)
        scalar_features = self.scalar_branch(scalars)
        features = self.bottleneck(torch.cat([curve_features, scalar_features], dim=1))
        return torch.sigmoid(self.mu_head(features)), self.log_sigma_head(features)


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[InverseInitializer, dict | None]:
    """Load an initializer checkpoint and its optional input-normalization data."""
    model = InverseInitializer().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    norm_stats = None
    if "norm_stats" in checkpoint:
        norm_stats = {key: np.array(value) for key, value in checkpoint["norm_stats"].items()}
    return model, norm_stats


def predict(
    model: InverseInitializer,
    t: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    codec: Any | None = None,
    norm_stats: dict | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict parameter means and standard deviations in the physical domain.

    ``ParamCodec`` is imported only when needed, so this optional accelerator can
    be imported without loading the full PyBaMM calibration stack.
    """
    if codec is None:
        from .param_codec import ParamCodec

        codec = ParamCodec.from_defaults()

    curves, scalar_dict = build_feature_matrix(t, v, temperature)
    scalars = np.array([scalar_dict[key] for key in SCALAR_KEYS], dtype=np.float32)
    if norm_stats is not None:
        curves = (curves - norm_stats["curves_mean"].squeeze(0)) / norm_stats["curves_std"].squeeze(0)
        scalars = (scalars - norm_stats["scalars_mean"].squeeze(0)) / norm_stats["scalars_std"].squeeze(0)

    curve_tensor = torch.from_numpy(curves[np.newaxis].astype(np.float32)).to(device)
    scalar_tensor = torch.from_numpy(scalars[np.newaxis].astype(np.float32)).to(device)
    with torch.no_grad():
        mu_norm, log_sigma = model(curve_tensor, scalar_tensor)

    mu_norm = mu_norm[0].cpu().numpy()
    sigma_norm = np.exp(np.clip(log_sigma[0].cpu().numpy(), -4.0, 4.0))
    mu = codec.denormalize(mu_norm)
    bounds = np.asarray(codec.bounds, dtype=float)
    return mu, sigma_norm * (bounds[:, 1] - bounds[:, 0])


def generate_warm_start_population(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_individuals: int,
    bounds: list[tuple[float, float]],
    k: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Sample a bounded differential-evolution population around ``mu``."""
    rng = np.random.default_rng(seed)
    population = np.zeros((n_individuals, len(mu)))
    population[0] = mu
    for index in range(1, n_individuals):
        population[index] = rng.normal(mu, k * sigma)
    bounds_array = np.asarray(bounds, dtype=float)
    return np.clip(population, bounds_array[:, 0], bounds_array[:, 1])
