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
    n_runs: int = 5


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
    scheduler = "ReduceLROnPlateau"
    schedulerparams = dict(mode="min", patience=500, factor=0.1, min_lr=1e-5)
    scheduler_warmup_phase2 = 500

    # -------------------------
    # Phase control
    # -------------------------
    phase1_epochs = 30000
    phase2_epochs = 30000
    phase3_epochs = 10000
    print_every = 100

    # ==========================================================
    # Imaginary-weights anneal + plateau-reset (shared for phases 1/2)
    # ==========================================================
    imag_weights_penalty_enabled_phase1 = True
    imag_weights_penalty_enabled_phase2 = True
    imag_weights_penalty_enabled_phase3 = True

    imag_anneal_epochs = 5000
    imag_w_coeff_start = 1e-4
    imag_w_coeff_end = 1e3
    imag_anneal_mode = "exp"  # "linear" or "exp"

    # plateau logic: after anneal in each cycle, monitor data loss
    imag_plateau_patience = 1000
    imag_plateau_rel_tol = 1e-4 # 1e-6
    imag_plateau_check_after_anneal = True  # start plateau counting only after anneal is finished in the cycle
    imag_plateau_min_epoch_in_phase = 10000     # additional warmup in each phase before plateau logic starts

    # when plateau triggers: reinit Im(weights) and restart anneal cycle
    # imag_reinit_scale = 1.0  # Im ~ U(-scale/2, scale/2) for active weights
    imag_reinit_gain = 1.0
    imag_reinit_scale_min = 1e-4
    imag_reinit_scale_max = 0.1

    # ==========================================================
    # Sparsification (Phase 2 only): L1L0 on REAL(weights) only
    # ==========================================================
    l1l0_enabled_phase1 = False
    l1l0_enabled_phase2 = True
    l1l0_enabled_phase3 = False

    l1l0_on_real_only = False #True
    l1l0_real_reg_coeff_phase1 = 1e-3
    l1l0_real_reg_coeff_phase2 = 1e3

    # L1L0 schedule parameters (used when enabled)
    alpha_start = 1e-1
    alpha_end = 1e-1
    l1l0_start_epoch = 0
    l1l0_end_epoch = phase2_epochs
    l1l0_s = 0.001
    l1l0_eps = 1e-12

    # ==========================================================
    # Pruning (Phase 2 only, as per your strategy)
    # ==========================================================
    pruning_enabled_phase1 = False
    pruning_enabled_phase2 = True
    pruning_enabled_phase3 = False

    pruning_start_epoch = 5000
    pruning_period = 1000
    pruning_threshold = 1e-2

    # ==========================================================
    # Phase 3: keep imag penalty "big" (fixed), no anneal/reset, no sparsity/pruning
    # ==========================================================
    imag_w_coeff_phase3 = 1e3 #imag_w_coeff_end

    # division normalization
    normalize_divisions = True
    normalize_divisions_eps = 1e-12
    
    


class CEQLConfig:
    device = CEQLModelTrainingConfig.device
    no_params_list = [
        [
            {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'const',  'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'const',  'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'square', 'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'square', 'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'sqrt',   'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'sqrt',   'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'exp',   'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'exp',   'library_function_type': 'unary',  'in_channels': 1},
            {"library_function": 'sin', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {"library_function": 'sin', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {"library_function": 'cos', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {"library_function": 'cos', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {'library_function': 'log',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-100},
            {'library_function': 'log',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-100},
            {'library_function': 'tan',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-4},
            {'library_function': 'tan',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-4},
            {'library_function': "tanh",   'library_function_type': 'unary', "in_channels": 1, 'stair_step_size': 1e-4},
            {'library_function': "tanh",   'library_function_type': 'unary', "in_channels": 1, 'stair_step_size': 1e-4},
            {'library_function': 'mul',    'library_function_type': 'binary', 'in_channels': 2},
            {'library_function': 'div',    'library_function_type': 'binary', 'in_channels': 2, 'stair_step_size': 1e-4},
        ],
        [
            {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'const',  'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'const',  'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'square', 'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'square', 'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'sqrt',   'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'sqrt',   'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'exp',   'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'exp',   'library_function_type': 'unary',  'in_channels': 1},
            {"library_function": 'sin', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {"library_function": 'sin', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {"library_function": 'cos', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {"library_function": 'cos', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {'library_function': 'log',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-100},
            {'library_function': 'log',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-100},
            {'library_function': 'tan',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-4},
            {'library_function': 'tan',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-4},
            {'library_function': "tanh",   'library_function_type': 'unary', "in_channels": 1, 'stair_step_size': 1e-4},
            {'library_function': "tanh",   'library_function_type': 'unary', "in_channels": 1, 'stair_step_size': 1e-4},
            {'library_function': 'mul',    'library_function_type': 'binary', 'in_channels': 2},
            {'library_function': 'div',    'library_function_type': 'binary', 'in_channels': 2, 'stair_step_size': 1e-4},
        ],
        [
            {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'const',  'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'const',  'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'square', 'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'square', 'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'sqrt',   'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'sqrt',   'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'exp',   'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'exp',   'library_function_type': 'unary',  'in_channels': 1},
            {"library_function": 'sin', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {"library_function": 'sin', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {"library_function": 'cos', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {"library_function": 'cos', 'library_function_type': 'unary', 'in_channels': 1, 'damp_gamma': 1.0, 'damp_p': 1.0},
            {'library_function': 'log',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-100},
            {'library_function': 'log',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-100},
            {'library_function': 'tan',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-4},
            {'library_function': 'tan',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-4},
            {'library_function': "tanh",   'library_function_type': 'unary', "in_channels": 1, 'stair_step_size': 1e-4},
            {'library_function': "tanh",   'library_function_type': 'unary', "in_channels": 1, 'stair_step_size': 1e-4},
            {'library_function': 'mul',    'library_function_type': 'binary', 'in_channels': 2},
            {'library_function': 'div',    'library_function_type': 'binary', 'in_channels': 2, 'stair_step_size': 1e-4},
        ],
    ]

    n_input_fields = 5
    n_symbolic_layers = len(no_params_list)

    functions_dict = {
        'x':      sp.Symbol('x'),
        'id':     lambda x: x,
        'const':  lambda x: sp.Integer(1),
        'square': lambda x: x**2,
        'cube':   lambda x: x**3,
        'sqrt':   lambda x: sp.sqrt(x),
        'exp':    lambda x: sp.exp(x),
        'sin':    lambda x: sp.sin(x),
        'cos':    lambda x: sp.cos(x),
        'tan':    lambda x: sp.tan(x),
        'tanh':   lambda x: sp.tanh(x),
        'log':    lambda x: sp.log(x),
        'mul':    lambda a, b: a * b,
        'div':    lambda a, b: a / b,
        'pow':    lambda a, b: a**b,
    }


BENCH = KornsBenchmarkConfig()
PYSR = PySRConfig()
CEQL_TRAIN = CEQLModelTrainingConfig()
CEQL = CEQLConfig()