from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List


FEATURE_NAMES: List[str] = ["x0", "x1", "x2", "x3", "x4"]


@dataclass(frozen=True)
class KornsBenchmarkConfig:
    hdf5_path: str = "korns_dataset.hdf5"
    # results_csv_path: str = "korns_benchmark_results.csv"
    test_size: float = 0.1
    split_seed: int = 0
    per_problem_seed_offset: int = 1000
    algo_seed_offset: int = 10_000
    run_seed_offset: int = 1_000_000
    n_runs: int = 5


@dataclass(frozen=True)
class PySRConfig:
    niterations: int = 100
    populations: int = 100
    maxsize: int = 20
    timeout_in_seconds: Optional[int] = None
    unary_ops: List[str] = ("sin", "cos", "tan", "tanh", "exp", "log", "sqrt")
    binary_ops: List[str] = ("+", "-", "*", "/", "^")
    verbosity: int = 0
    progress: bool = False
    temp_equation_file: bool = True
    delete_tempfiles: bool = True
    model_selection: str = "best"
    elementwise_loss: str = "loss(x, y) = (x - y)^2"
    results_csv_path: str = "korns_pysr_benchmark_results.csv"


BENCH = KornsBenchmarkConfig()
PYSR = PySRConfig()