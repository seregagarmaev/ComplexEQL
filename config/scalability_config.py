from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    seed: int = 0

    # Polynomial uses first k variables; d can be larger (nuisance dims)
    k: int = 2

    train_domain_min: float = -5.0
    train_domain_max: float = 5.0
    test_domain_min_abs: float = 5.0
    test_domain_max_abs: float = 10.0

    n_train: int = 10_000
    n_test: int = 10_000
    d_list: tuple[int, ...] = (2, 4, 8, 16, 32, 64)

    out_path: str = "scalability_experiment_data.h5"


@dataclass(frozen=True)
class OperonConfig:
    # Experiment scope
    kinds: tuple[str, ...] = ("P3",)

    # If len(seeds) < runs_per_d, the script will fall back to range(runs_per_d)
    runs_per_d: int = 5
    seeds: tuple[int, ...] = (0,)

    # String extraction
    sym_decimals: int = 15

    # Reporting
    report_csv_path: str = "reports/operon_scalability.csv"

    # Plotting
    y_log_scale: bool = True
    x_label: str = "Number of Input Variables"
    y_label: str = "MSE"

    # Operator library
    allowed_symbols: str = "add,sub,mul,square,constant,variable"

    # Evolution / search parameters
    generations: int = 100
    population_size: int = 100
    max_length: int = 20
    max_depth: int = 10

    # Local optimization
    optimizer: str = "lm"
    optimizer_iterations: int = 50
    local_search_probability: float = 0.05
    lamarckian_probability: float = 0.5

    # Objectives and selection
    objectives: tuple[str, ...] = ("mse", "length")
    model_selection_criterion: str = "mean_squared_error"

    # Logging
    verbose: int = 1
    print_best_expression: bool = True


DataCFG = DataConfig()
OperonCFG = OperonConfig()