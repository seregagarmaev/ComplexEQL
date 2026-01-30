from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DataCFG:
    seed: int = 0

    n_train: int = 4096
    n_test_interp: int = 8192
    n_test_extrap: int = 8192

    train_low: float = -2.0
    train_high: float = 2.0

    extrap_a_low: float = -4.0
    extrap_a_high: float = -2.0
    extrap_b_low: float = 2.0
    extrap_b_high: float = 4.0

    noise_sigma_rel: float = 0.0

    coeff_min_abs: float = 0.5
    coeff_max_abs: float = 3.0
    coeff_decimals: int = 2

    inner_const_min_abs: float = 0.5
    inner_const_max_abs: float = 3.0
    inner_const_decimals: int = 2

    batch_size: int = 8192
    max_rounds: int = 200

    grid_n: int = 81
    pole_eps: float = 1e-3
    no_pole_delta: float = 2e-1
    max_rational_trials: int = 20000

    h5_path: str = "data/sr_benchmark.h5"