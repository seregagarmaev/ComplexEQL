from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import sympy as sp


@dataclass(frozen=True)
class EmpiricalBenchConfig:
    # results_dir: str = "reports/empiricalbench_ceql"
    results_csv_path: str = "reports/empiricalbench_ceql_results.csv"

    keys: tuple[str, ...] = (
        "hubble",
        "kepler",
        "newton",
        "planck",
        "leavitt",
        "schechter",
        "bode",
        "ideal_gas",
        "rydberg",
    )

    n_runs: int = 1
    base_seed: int = 0

    test_size: float = 0.2
    split_seed: int = 0


class CEQLModelTrainingConfig:
    device = "cpu"
    loss_function = "MSELoss"
    train_batch_size = 100

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

    phase1_epochs = 10000
    l1_reg_coeff_phase1 = 1e-10
    imag_w_coeff_phase1 = 1e-10
    phase1_prune_enabled = True
    phase1_prune_threshold = 1e-3

    phase2_epochs = 20000
    l1_reg_coeff_phase2 = 1e-3
    imag_w_coeff_phase2 = 1e-10

    pruning_fraction_phase2 = 0.3
    pruning_threshold_min = 1e-3
    pruning_threshold_max = 100.0
    pruning_min_edges_per_layer = 10
    phase2_prune_warmup_epochs = 0
    prune_every_epochs = 20000

    normalize_divisions_during_phase2 = True

    phase3_epochs = 10000
    l1_reg_coeff_phase3 = 1e-7
    imag_w_coeff_phase3 = 1e-3


class CEQLConfig:
    device = CEQLModelTrainingConfig.device

    no_params_list = [
        [
            {"op": "id",    "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "square","type": "unary"},
            {"op": "sqrt",  "type": "unary"},
            {"op": "exp",   "type": "unary"},
            {"op": "log",   "type": "unary"},
            {"op": "mul",   "type": "binary"},
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
        "mul":    lambda a, b: a * b,
        "div":    lambda a, b: a / b,
    }


EBENCH = EmpiricalBenchConfig()
CEQL_TRAIN = CEQLModelTrainingConfig()
CEQL = CEQLConfig()
