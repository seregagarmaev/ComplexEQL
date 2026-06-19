from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import sympy as sp


@dataclass(frozen=True)
class DataCFG:
    h5_path: str = "data/ceql_ablation_3var_linear_rational.h5"


@dataclass(frozen=True)
class AblationRunConfig:
    results_path: str = "reports/ceql_ablation_3var_linear_rational.csv"

    n_runs: int = 5
    base_seed: int = 0

    benchmark_group_indices: tuple[int, ...] = (0,)

    ablations: Tuple[str, ...] = (
        "ceql_original",
        "ceql_original_with_log",
        "ceql_real_weights",
        "ceql_no_imaginary_penalty",
        "ceql_no_skip_connections",
    )


class CEQLModelTrainingConfig:
    device = "cpu"
    loss_function = "MSELoss"
    train_batch_size = 2**14

    lr = 1e-3
    scheduler = "ReduceLROnPlateau"
    schedulerparams = dict(mode="min", patience=2000, factor=0.1, min_lr=1e-5)

    phase1_epochs = 100000
    phase2_epochs = 250000
    phase3_epochs = 50000
    print_every = 1000

    l1_reg_coeff_phase1 = 1e-10
    l1_reg_coeff_phase2 = 1e-9
    l1_reg_coeff_phase3 = 0.0
    l1_on_real_only = False
    l1_eps = 1e-12
    phase3_l1_enabled = True

    phase1_prune_enabled = False
    phase1_prune_threshold = 1e-3
    pruning_fraction_phase2 = 0.2
    pruning_threshold_min = 0.0
    pruning_threshold_max = 1e-2
    pruning_min_edges_per_layer = 5
    phase2_prune_warmup_epochs = 0

    normalize_divisions_phase2 = False
    normalize_divisions_phase3 = True
    normalize_divisions_eps = 1e-12

    imag_w_coeff_phase1 = 1e-10
    imag_w_coeff_phase2 = 1e-5
    imag_w_coeff_phase3 = 1e-2

    phase1_imag_shrink_enabled = False
    phase2_imag_shrink_enabled = False
    phase3_imag_shrink_enabled = False

    phase1_imag_shrink_coeff = 1.0
    phase2_imag_shrink_coeff = 1.0
    phase3_imag_shrink_coeff = 0.99
    phase3_force_real = False

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

    ablation_prune_enabled = True
    ablation_prune_fraction_phase2 = 0.1
    ablation_min_edges_total = 20
    prune_every_epochs = 10000


class CEQLConfig:
    device = CEQLModelTrainingConfig.device

    use_skip_connections = True
    weight_domain = "complex"

    no_params_list = [
        [
            {"op": "id", "type": "unary"},
            {"op": "id", "type": "unary"},
            {"op": "id", "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "square", "type": "unary"},
            {"op": "square", "type": "unary"},
            {"op": "div", "type": "binary"},
            {"op": "div", "type": "binary"},
        ],
        [
            {"op": "id", "type": "unary"},
            {"op": "id", "type": "unary"},
            {"op": "id", "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "square", "type": "unary"},
            {"op": "square", "type": "unary"},
            {"op": "div", "type": "binary"},
            {"op": "div", "type": "binary"},
        ],
    ]

    n_input_fields = 3
    n_symbolic_layers = len(no_params_list)

    functions_dict = {
        "x": sp.Symbol("x"),
        "id": lambda x: x,
        "const": lambda x: sp.Integer(1),
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
        "sqrt": lambda x: sp.sqrt(x),
        "exp": lambda x: sp.exp(x),
        "log": lambda x: sp.log(x),
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "x^y": lambda a, b: a**b,
        "resonator": lambda a, b: sp.Abs(sp.re(a / b)),
    }


DATA = DataCFG()
ABLATION = AblationRunConfig()
CEQL_TRAIN = CEQLModelTrainingConfig()
CEQL = CEQLConfig()