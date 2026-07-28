"""Feature extraction for the optional inverse calibration initializer."""

import numpy as np


def resample_curve(t: np.ndarray, y: np.ndarray, n_points: int = 256) -> np.ndarray:
    """Linearly resample a time series to ``n_points`` uniformly spaced samples."""
    t_uniform = np.linspace(float(t[0]), float(t[-1]), n_points)
    return np.interp(t_uniform, t, y)


def compute_dvdt(t: np.ndarray, v: np.ndarray, n_points: int = 256) -> np.ndarray:
    """Resample a voltage curve and compute its time derivative."""
    t_uniform = np.linspace(float(t[0]), float(t[-1]), n_points)
    v_resampled = np.interp(t_uniform, t, v)
    return np.gradient(v_resampled, t_uniform)


def build_feature_matrix(
    t: np.ndarray,
    v: np.ndarray,
    temperature: np.ndarray,
    n_points: int = 256,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build the inverse initializer's three curve and five scalar features.

    The curve channels are voltage, ``dV/dt``, and temperature.  The scalar
    features describe the cutoff time, voltage endpoints, and temperature range.
    """
    v_resampled = resample_curve(t, v, n_points)
    dvdt = compute_dvdt(t, v, n_points)
    temperature_resampled = resample_curve(t, temperature, n_points)
    curves = np.stack([v_resampled, dvdt, temperature_resampled], axis=0)
    scalars = {
        "t_cutoff": float(t[-1]),
        "V0": float(v[0]),
        "Vend": float(v[-1]),
        "Tmax": float(np.max(temperature)),
        "ambient_T": float(temperature[0]),
    }
    return curves, scalars
