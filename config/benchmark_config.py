from __future__ import annotations
from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class DataCFG:
    seed: int = 0

    n_train: int = 1024  # 4096
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
    loss_function = "L1Loss" # "MSELoss"
    train_batch_size = 1024 #2**14

    lr = 1e-3
    scheduler = "ReduceLROnPlateau"
    schedulerparams = dict(mode="min", patience=2000, factor=0.1, min_lr=1e-6)

    print_every = 1000

    l1_on_real_only = False
    l1_eps = 1e-12

    normalize_divisions_eps = 1e-12

    clamp_pred = True
    clamp_limit = 1e10
    pred_abs_max = 1e18

    use_op_params = False
    op_param_schedules = {}

    phase1_epochs = 30000
    l1_reg_coeff_phase1 = 1e-10
    imag_w_coeff_phase1 = 1e-10
    phase1_prune_enabled = True
    phase1_prune_threshold = 1e-3

    phase2_epochs = 100000
    l1_reg_coeff_phase2 = 1e-3
    imag_w_coeff_phase2 = 1e3

    pruning_fraction_phase2 = 0.2
    pruning_threshold_min = 1e-10
    pruning_threshold_max = 100.0
    pruning_min_edges_per_layer = 12
    phase2_prune_warmup_epochs = 0
    prune_every_epochs = 10000

    normalize_divisions_phase2 = True
    normalize_divisions_phase3 = True

    phase3_epochs = 20000
    phase3_l1_enabled = True #False
    l1_reg_coeff_phase3 = 1e-7
    imag_w_coeff_phase3 = 0 # 1e-3
    phase3_imag_shrink_enabled = False
    phase3_imag_shrink_coeff = 1.0 # 0.99
    phase3_force_real = True


class CEQLConfig:
    device = CEQLModelTrainingConfig.device

    no_params_list = [
        [
            # {"op": "id",    "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "square","type": "unary"},
            {"op": "square","type": "unary"},
            # {"op": "log",   "type": "unary"},
            # {"op": "log",   "type": "unary"},
            # {"op": "sqrt",   "type": "unary"},
            {"op": "mul",   "type": "binary"},
            {"op": "mul",   "type": "binary"},
            # {"op": "div",   "type": "binary"},
            # {"op": "x^y",   "type": "binary"},
        ],
        [
            # {"op": "id",    "type": "unary"},
            # {"op": "id",    "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "square","type": "unary"},
            {"op": "log",   "type": "unary"},
            {"op": "sqrt",   "type": "unary"},
            {"op": "mul",   "type": "binary"},
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
        "log":    lambda x: sp.log(x + 1.0),
        # "log":    lambda x: sp.log(x),
        "log10":  lambda x: sp.log(x + 1.0, 10),
        "mul":    lambda a, b: a * b,
        "div":    lambda a, b: a / b,
        "x^y":    lambda a, b: a ** b,
    }


CEQL_TRAIN = CEQLModelTrainingConfig()
CEQL = CEQLConfig()