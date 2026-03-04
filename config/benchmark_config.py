from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple

import sympy as sp


@dataclass(frozen=True)
class DataCFG:
    seed: int = 0

    n_train: int = 128 #4096
    n_test_interp: int = 8192
    n_test_extrap: int = 8192

    train_low: float = -2.0
    train_high: float = 2.0

    extrap_a_low: float = -4.0
    extrap_a_high: float = -2.0
    extrap_b_low: float = 2.0
    extrap_b_high: float = 4.0

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


class CEQLModelTrainingConfig:
    results_path = "reports/sr_benchmark_ceql.csv"
    device = "cpu"
    loss_function = "MSELoss"
    train_batch_size = 2**14

    lr = 1e-2
    scheduler = "ReduceLROnPlateau"
    schedulerparams = dict(mode="min", patience=2000, factor=0.1, min_lr=1e-5)

    phase1_epochs = 100000
    phase2_epochs = 200000 #100000
    phase3_epochs = 50000
    print_every = 1000

    l1_reg_coeff_phase1 = 1e-10
    l1_reg_coeff_phase2 = 1e-7
    l1_reg_coeff_phase3 = 1e-7
    l1_on_real_only = False
    l1_eps = 1e-12
    phase3_l1_enabled = True #False

    phase1_prune_enabled = False #True
    phase1_prune_threshold = 1e-3
    pruning_fraction_phase2 = 0.2
    pruning_threshold_min = 0 #1e-3
    pruning_threshold_max = 1e-2
    pruning_min_edges_per_layer = 5
    phase2_prune_warmup_epochs = 0

    normalize_divisions_phase2 = False
    normalize_divisions_phase3 = True # False
    normalize_divisions_eps = 1e-12

    imag_w_coeff_phase1 = 1e-10
    imag_w_coeff_phase2 = 1e-3
    imag_w_coeff_phase3 = 1e3

    phase1_imag_shrink_enabled = False
    phase2_imag_shrink_enabled = False #True
    phase3_imag_shrink_enabled = False
    
    phase1_imag_shrink_coeff = 1.0
    phase2_imag_shrink_coeff = 1.0
    phase3_imag_shrink_coeff = 0.99
    phase3_force_real = False #True

    theta_eps = 1e-12
    theta_coeff_phase1 = 1e-10
    theta_coeff_phase2 = 1e3
    theta_coeff_phase3 = 1e3

    clamp_pred = True
    clamp_limit = 1e10
    pred_abs_max = 1e20

    theta_ops = ("log", "sqrt")
    use_op_params = False
    op_param_schedules = {}

    # importance_prune_enabled = True
    # importance_prune_fraction_phase1_end = 0.2
    # importance_prune_fraction_phase2 = 0.2
    # importance_prune_fraction_phase3_end = 0.0
    # importance_min_edges_total = 10

    ablation_prune_enabled: bool = True
    ablation_prune_fraction_phase2: float = 0.1
    ablation_min_edges_total: int = 15
    prune_every_epochs = 10000



class CEQLConfig:
    device = CEQLModelTrainingConfig.device

    no_params_list = [
        [
            # {"op": "id",    "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "square","type": "unary"},
            {"op": "square","type": "unary"},
            # {"op": "log",   "type": "unary"},
            # {"op": "log",   "type": "unary"},
            # {"op": "sqrt",   "type": "unary"},
            # {"op": "sqrt",   "type": "unary"},
            {"op": "mul",   "type": "binary"},
            {"op": "mul",   "type": "binary"},
            # {"op": "div",   "type": "binary"},
            # {"op": "x^y",   "type": "binary"},
        ],
        [
            {"op": "id",    "type": "unary"},
            # {"op": "const", "type": "unary"},
            # {"op": "square","type": "unary"},
            {"op": "log",   "type": "unary"},
            {"op": "log",   "type": "unary"},
            # {"op": "log",   "type": "unary"},
            {"op": "sqrt",   "type": "unary"},
            {"op": "sqrt",   "type": "unary"},
            # {"op": "sqrt",   "type": "unary"},
            # {"op": "mul",   "type": "binary"},
            {"op": "div",   "type": "binary"},
            {"op": "div",   "type": "binary"},
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
        # "log":    lambda x: sp.log(x + 1.0),
        "log":    lambda x: sp.log(x),
        "log10":  lambda x: sp.log(x + 1.0, 10),
        "mul":    lambda a, b: a * b,
        "div":    lambda a, b: a / b,
        "x^y":    lambda a, b: a ** b,
    }


@dataclass(frozen=True)
class PySRConfig:
    results_path: str = "reports/sr_benchmark_pysr.csv"

    n_runs: int = 5
    base_seed: int = 0

    niterations: int = 1000
    populations: int = 200
    maxsize: int = 20
    timeout_in_seconds: Optional[int] = None

    unary_ops: List[str] = ("log", "sqrt", "square")
    binary_ops: List[str] = ("+", "-", "*", "/")

    elementwise_loss: str = "loss(x, y) = (x - y)^2"
    model_selection: str = "best"

    verbosity: int = 0
    progress: bool = False
    temp_equation_file: bool = True
    delete_tempfiles: bool = True


@dataclass(frozen=True)
class SINDyConfig:
    results_path: str = "reports/sr_benchmark_sindy.csv"

    n_runs: int = 5
    base_seed: int = 0

    # feature library
    poly_degree: int = 2
    include_interaction: bool = True
    include_bias: bool = True
    max_library_features: int = 100

    unary_ops: List[str] = ("log", "sqrt")
    binary_ops: List[str] = ("*", "/")

    # sparse regression
    threshold: float = 1e-2
    alpha: float = 1e-6
    max_iter: int = 1000
    normalize_columns: bool = True

    # numerical stabilizers
    log_eps: float = 1e-12
    sqrt_abs: bool = True
    div_eps: float = 1e-6
    max_feature_abs: float = 1e6

    coef_zero_tol: float = 1e-12


@dataclass(frozen=True)
class EQLDivConfig:
    results_path: str = "reports/sr_benchmark_eql_div.csv"

    name: str = "eql_div"

    n_runs: int = 5
    base_seed: int = 0

    # architecture
    num_h_layers: int = 2
    layer_width: int = 5
    layer_ops: Tuple[str, ...] = ("id", "id", "log", "multiply", "multiply")
    out_op: str = "id"

    # training
    epoch_factor: int = 1000
    penalty_every: int = 10
    batch_size: int = 128
    penalty_examples_cap: int = 2048

    learning_rate: float = 1e-3
    beta1: float = 0.9

    # sparsity
    reg_sched: Tuple[float, float] = (0.1, 0.9)
    reg_scale: float = 1e-2
    l0_threshold: float = 1e-2

    # division
    test_div_threshold: float = 1e-2
    weight_init_param: float = 1.0
    output_bound: Optional[float] = None

    # symbolic
    complexity_threshold: float = 1e-3
    symbolic_prune_threshold: float = 1e-2
    round_decimals: int = 6
    simplify: bool = False


CEQL_TRAIN = CEQLModelTrainingConfig()
CEQL = CEQLConfig()
PYSR = PySRConfig()
SINDY = SINDyConfig()
EQLDIV = EQLDivConfig()