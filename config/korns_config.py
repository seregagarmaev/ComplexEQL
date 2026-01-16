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

    print_every = 1000

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
    clamp_limit = 1e6

    # -------------------------
    # Trig control (r is driven by the cycle logic; schedules here are unused)
    # -------------------------
    use_op_params = True
    op_param_schedules = {}  # r is set via build_trig_op_params(cfg, r_value)

    # =========================================================
    # Cycle-based training strategy
    # =========================================================

    # ---- Cycle stage A: ramp r from r_start_cycle -> r_end_cycle (log), no sparsity
    cycle_ramp_epochs = 5000
    r_start_cycle = 0.01
    r_end_cycle = 1.0

    # ---- Cycle stage B: r fixed at 1.0, sparsity ON, division normalization after each epoch
    cycle_sparsity_epochs = 20000
    l1_reg_coeff_cycle = 1e-3 # applied to whole complex number, not real only. TODO: rename
    normalize_divisions_during_sparsity = True

    # ---- End-of-cycle pruning
    pruning_fraction_cycle = 0.5
    pruning_threshold_min = 1e-2
    pruning_threshold_max = 1e0
    pruning_min_edges_per_layer = 10

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
    imag_w_coeff_post = 1e3
    
    


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
            {"op": "tan",   "type": "unary"},
            {"op": "tan",   "type": "unary"},
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
            {"op": "mul",   "type": "binary"},
            {"op": "div",   "type": "binary"},
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


@dataclass(frozen=True)
class SINDyConfig:
    # Use the same operator sets as PySR
    unary_ops: List[str] = ("sin", "cos", "tan", "tanh", "exp", "log", "sqrt")
    binary_ops: List[str] = ("+", "-", "*", "/", "^")

    # Polynomial library controls * and ^ (powers) coverage
    poly_degree: int = 2
    include_interaction: bool = True
    include_bias: bool = True
    max_library_features: int = 30

    # Sparse optimizer (STLSQ)
    threshold: float = 1e-2
    alpha: float = 1e-6
    max_iter: int = 50
    normalize_columns: bool = True
    stlsq_verbose: bool = True

    # Numerics for unary ops
    clip_exp: float = 50.0
    log_eps: float = 1e-12
    sqrt_abs: bool = True

    # Numerics for division
    div_eps: float = 1e-6

    # Stabilize Theta(X)
    max_feature_abs: float = 1e6
    drop_nonfinite_rows: bool = True

    # Readout / logging
    coef_zero_tol: float = 1e-12
    results_csv_path: str = "korns_sindy_benchmark_results.csv"


@dataclass(frozen=True)
class EQLDivConfig:
    # Naming / output
    name: str = "eql_div"
    results_csv_path: str = "korns_eql_div_benchmark_results.csv"

    # Training
    epoch_factor: int = 1000
    penalty_every: int = 10
    batch_size: int = 512
    penalty_examples_cap: int = 2048

    # Architecture
    num_h_layers: int = 2
    layer_width: int = 1

    # Only ops implemented in EQL_Layer_tf AND numerically safe by default
    # (log/exp are implemented but unsafe -> excluded)
    layer_ops: Tuple[str, ...] = ("id", "sin", "cos", "exp", "log", "multiply")

    # Output op
    out_op: str = "reg_div"

    # Optimizer
    learning_rate: float = 1e-3
    beta1: float = 0.9

    # Sparsity
    reg_sched: Tuple[float, float] = (0.1, 0.9)
    reg_scale: float = 1e-2
    l0_threshold: float = 1e-2

    # Division / bounds
    test_div_threshold: float = 1e-2
    weight_init_param: float = 1.0
    output_bound: Optional[float] = None

    # Symbolic / reporting
    complexity_threshold: float = 1e-3
    symbolic_prune_threshold: float = 1e-2
    round_decimals: int = 6
    simplify: bool = False


EQLDIV = EQLDivConfig()
SINDY = SINDyConfig()
BENCH = KornsBenchmarkConfig()
PYSR = PySRConfig()
CEQL_TRAIN = CEQLModelTrainingConfig()
CEQL = CEQLConfig()