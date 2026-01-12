from __future__ import annotations

from dataclasses import dataclass
import sympy as sp
from typing import Optional, List


FEATURE_NAMES: List[str] = ["x0", "x1", "x2", "x3", "x4"]


@dataclass(frozen=True)
class KornsBenchmarkConfig:
    hdf5_path: str = "korns_dataset_train_extrap.hdf5"
    # results_csv_path: str = "korns_benchmark_results.csv"
    test_size: float = 0.1
    split_seed: int = 0
    per_problem_seed_offset: int = 1000
    algo_seed_offset: int = 10_000
    run_seed_offset: int = 1_000_000
    n_runs: int = 1 #5


@dataclass(frozen=True)
class PySRConfig:
    niterations: int = 1000
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


class CEQLModelTrainingConfig:
    device = "cpu"
    loss_function = "MSELoss"
    train_batch_size = 2**14

    # -------------------------
    # Optimization
    # -------------------------
    lr = 1e-3
    scheduler = "ReduceLROnPlateau"   # used ONLY in the final 10000 epochs
    schedulerparams = dict(mode="min", patience=1000, factor=0.1, min_lr=1e-8)

    print_every = 500

    # -------------------------
    # L1 sparsity
    # -------------------------
    l1_on_real_only = False    # if True: penalize only Re(weights)
    l1_eps = 1e-12             # only used for magnitude stability when complex

    # -------------------------
    # Division normalization
    # -------------------------
    normalize_divisions_eps = 1e-12  # used during sparsity stage inside cycles

    # -------------------------
    # Optional clamp (kept from utils behavior)
    # -------------------------
    clamp_pred = True
    clamp_limit = 1e2

    # -------------------------
    # Trig control (r is driven by the cycle logic; schedules here are unused)
    # -------------------------
    use_op_params = True
    op_param_schedules = {}  # r is set via build_trig_op_params(cfg, r_value)

    # =========================================================
    # Cycle-based training strategy
    # =========================================================

    # ---- Cycle stage A: ramp r from r_start_cycle -> r_end_cycle (log), no sparsity
    cycle_ramp_epochs = 5000 #10000
    r_start_cycle = 0.01
    r_end_cycle = 1.0

    # ---- Cycle stage B: r fixed at 1.0, sparsity ON, division normalization after each epoch
    cycle_sparsity_epochs = 5000 #10000
    l1_reg_coeff_cycle = 1e-3 # applied to whole complex number, not real only. TODO: rename
    normalize_divisions_during_sparsity = True

    # ---- End-of-cycle pruning
    pruning_fraction_cycle = 0.2
    pruning_min_edges_total = 20

    # ---- Small imag(weights) penalty applied throughout cycles
    imag_w_coeff_cycle = 1e-3

    # =========================================================
    # Post-cycle finishing strategy
    # =========================================================

    # 1) sparsity OFF, imag penalty ON, ramp r from r_start_post -> r_end_post (log)
    post_ramp_epochs = 10000
    r_start_post = 0.01
    r_end_post = 1.0

    # 2) r fixed at 1.0 for final optimization; scheduler ON here
    post_finetune_epochs = 10000

    # imag penalty during post stages (keep small)
    imag_w_coeff_post = 1e-3
    
    


class CEQLConfig:
    device = CEQLModelTrainingConfig.device
    no_params_list = [
        [
            {"op": "id",    "type": "unary"},
            {"op": "id",    "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "square","type": "unary"},
            {"op": "sqrt",  "type": "unary"},
            {"op": "exp",   "type": "unary"},
            {"op": "sin",   "type": "unary"},
            {"op": "cos",   "type": "unary"},
            {"op": "log",   "type": "unary"},
            {"op": "tan",   "type": "unary"},
            {"op": "tanh",  "type": "unary"},
            {"op": "mul",   "type": "binary"},
            {"op": "div",   "type": "binary"},
        ],
        [
            {"op": "id",    "type": "unary"},
            {"op": "id",    "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "square","type": "unary"},
            {"op": "sqrt",  "type": "unary"},
            {"op": "exp",   "type": "unary"},
            {"op": "sin",   "type": "unary"},
            {"op": "cos",   "type": "unary"},
            {"op": "log",   "type": "unary"},
            {"op": "tan",   "type": "unary"},
            {"op": "tanh",  "type": "unary"},
            {"op": "mul",   "type": "binary"},
            {"op": "div",   "type": "binary"},
        ],
        [
            {"op": "id",    "type": "unary"},
            {"op": "id",    "type": "unary"},
            {"op": "const", "type": "unary"},
            {"op": "square","type": "unary"},
            {"op": "sqrt",  "type": "unary"},
            {"op": "exp",   "type": "unary"},
            {"op": "sin",   "type": "unary"},
            {"op": "cos",   "type": "unary"},
            {"op": "log",   "type": "unary"},
            {"op": "tan",   "type": "unary"},
            {"op": "tanh",  "type": "unary"},
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
        "sin":    lambda x: sp.sin(x),
        "cos":    lambda x: sp.cos(x),
        "tan":    lambda x: sp.tan(x),
        "tanh":   lambda x: sp.tanh(x),
        "log":    lambda x: sp.log(x),
        "mul":    lambda a, b: a * b,
        "div":    lambda a, b: a / b,
    }


BENCH = KornsBenchmarkConfig()
PYSR = PySRConfig()
CEQL_TRAIN = CEQLModelTrainingConfig()
CEQL = CEQLConfig()