from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    # Reproducibility
    seed: int = 0

    # Expression family
    k: int = 2
    pole_domain: float = 5.0

    # Sampling domains
    train_domain_min: float = -5.0
    train_domain_max: float = 5.0
    test_domain_min_abs: float = 5.0
    test_domain_max_abs: float = 10.0

    # Dataset sizes and filtering
    delta0: float = 0.01
    n_train: int = 10_000
    n_test: int = 10_000
    d_list: tuple[int, ...] = (2, 4, 8, 16, 32, 64)

    # Output
    out_path: str = "scalability_experiment_data.h5"


DataCFG = DataConfig()
