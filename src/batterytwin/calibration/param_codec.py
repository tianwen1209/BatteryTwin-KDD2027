import json
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from .optimization import PARAM_ORDER, PARAM_BOUNDS_DEFAULT, build_param_update

@dataclass
class ParamCodec:
    param_names: List[str] = field(default_factory=lambda: list(PARAM_ORDER))
    bounds: List[Tuple[float, float]] = field(default_factory=lambda: list(PARAM_BOUNDS_DEFAULT))

    @property
    def n_dims(self) -> int:
        return len(self.param_names)

    @classmethod
    def from_defaults(cls) -> 'ParamCodec':
        return cls()

    @classmethod
    def from_config(cls, config_path: str=None) -> 'ParamCodec':
        if config_path is None:
            config_path = pathlib.Path(__file__).resolve().parents[3] / 'configs' / 'calibration.default.json'
        config_path = pathlib.Path(config_path)
        if not config_path.exists():
            return cls.from_defaults()
        with open(config_path, encoding='utf-8') as f:
            cfg = json.load(f)
        pb = cfg.get('param_bounds', {})
        names = list(PARAM_ORDER)
        bounds = []
        for i, name in enumerate(names):
            if name in pb:
                bounds.append((pb[name][0], pb[name][1]))
            else:
                bounds.append(PARAM_BOUNDS_DEFAULT[i])
        return cls(param_names=names, bounds=bounds)

    def decode(self, x: np.ndarray) -> dict:
        return {name: float(val) for name, val in zip(self.param_names, x)}

    def encode(self, param_dict: dict) -> np.ndarray:
        return np.array([param_dict[name] for name in self.param_names], dtype=float)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        span = hi - lo
        span[span == 0] = 1.0
        return (np.asarray(x) - lo) / span

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return lo + np.asarray(x_norm) * (hi - lo)

    def to_pybamm_update(self, x: np.ndarray, base_params) -> dict:
        return build_param_update(x, base_params)
