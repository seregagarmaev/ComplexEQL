from __future__ import annotations
from dataclasses import dataclass
import sympy as sp


@dataclass(frozen=True)
class DataConfig:
    seed: int = 0

    # Polynomial uses first k variables; d can be larger (nuisance dims)
    k: int = 2

    train_domain_min: float = -5.0
    train_domain_max: float = 5.0
    test_domain_min_abs: float = 5.0
    test_domain_max_abs: float = 10.0

    n_train: int = 1_000
    n_test: int = 1_000
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
    generations: int = 1000
    population_size: int = 1000
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


@dataclass
class CEQLModelTrainingConfig:
    device = "cpu"
    loss_function = "MSELoss"
    train_batch_size = 2**14
    report_csv_path: str = "reports/ceql_scalability.csv"

    lr = 1e-3
    scheduler = "ReduceLROnPlateau"
    schedulerparams = dict(mode="min", patience=2000, factor=0.1, min_lr=1e-6)

    print_every = 100

    l1_on_real_only = False #
    l1_eps = 1e-12

    normalize_divisions_eps = 1e-12

    clamp_pred = True
    clamp_limit = 1e10
    pred_abs_max = 1e18

    use_op_params = False
    op_param_schedules = {}

    # =========================================================
    # Phase-based training strategy
    # =========================================================

    # Phase 1: data + small L1(|w|) + small imag penalty
    phase1_epochs = 20000
    l1_reg_coeff_phase1 = 1e-3
    imag_w_coeff_phase1 = 1e1
    phase1_prune_enabled = True
    phase1_prune_threshold = 1e-1

    # Phase 2: higher sparsity + periodic pruning (pruning logic stays in utils.train)
    phase2_epochs = 30000
    l1_reg_coeff_phase2 = 1e-3
    imag_w_coeff_phase2 = 1e1

    pruning_fraction_phase2 = 0.3
    pruning_threshold_min = 1e-1
    pruning_threshold_max = 0.1
    pruning_min_edges_per_layer = 10
    phase2_prune_warmup_epochs = 0
    prune_every_epochs = 5000

    normalize_divisions_during_phase2 = True

    # Phase 3: sparsity OFF, imag penalty bigger, data fit
    phase3_epochs = 20000
    l1_reg_coeff_phase3 = 1e-7 #0.0
    imag_w_coeff_phase3 = 1e-3


@dataclass
class CEQLConfig:
    device = CEQLModelTrainingConfig.device

    no_params_list = [
        [
            {"op": "const",  "type": "unary"},
            {"op": "const",  "type": "unary"},
            {"op": "square", "type": "unary"},
            {"op": "square", "type": "unary"},
            {"op": "mul",    "type": "binary"},
            {"op": "mul",    "type": "binary"},
        ],
        [
            {"op": "id", "type": "unary"},
            {"op": "id", "type": "unary"},
            {"op": "const",  "type": "unary"},
            {"op": "const",  "type": "unary"},
            {"op": "square", "type": "unary"},
            {"op": "square", "type": "unary"},
            {"op": "mul",    "type": "binary"},
            {"op": "mul",    "type": "binary"},
        ],
    ]

    n_input_fields = 2
    n_symbolic_layers = len(no_params_list)

    functions_dict = {
        "x":      sp.Symbol("x"),
        "id":     lambda x: x,
        "const":  lambda x: sp.Integer(1),
        "square": lambda x: x**2,
        "cube":   lambda x: x**3,
        "sqrt":   lambda x: sp.sqrt(x),
        "exp":    lambda x: sp.exp(x),
        "log":    lambda x: sp.log(x + 1.0),
        "mul":    lambda a, b: a * b,
        "div":    lambda a, b: a / b,
    }


DataCFG = DataConfig()
OperonCFG = OperonConfig()
CEQL_TRAIN = CEQLModelTrainingConfig()
CEQL = CEQLConfig()