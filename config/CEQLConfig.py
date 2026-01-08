import sympy as sp


class ModelTrainingConfig:
    device = "cpu"
    loss_function = "MSELoss"
    train_batch_size = 2**14

    # -------------------------
    # Optimization
    # -------------------------
    lr = 1e-3
    use_scheduler_phase3 = True
    scheduler = "ReduceLROnPlateau"   # or None
    schedulerparams = dict(mode="min", patience=1000, factor=0.1, min_lr=1e-8)

    # -------------------------
    # -------------------------
    # Phase control
    # -------------------------
    phase1_epochs = 10000
    phase2_epochs = 10000
    phase3_epochs = 10000
    print_every = 100

    # -------------------------
    # L1L0 (constant alpha)
    # -------------------------
    l1l0_alpha = 1e-1
    l1l0_s = 0.001
    l1l0_eps = 1e-12

    l1l0_on_real_only = False

    l1l0_enabled_phase1 = False
    l1l0_enabled_phase2 = True
    l1l0_enabled_phase3 = False

    l1l0_real_reg_coeff_phase1 = 1e-3
    l1l0_real_reg_coeff_phase2 = 1e-1
    l1l0_real_reg_coeff_phase3 = 0.0

    # -------------------------
    # Imag penalty (constant per phase)
    # -------------------------
    imag_weights_penalty_enabled_phase1 = True
    imag_weights_penalty_enabled_phase2 = True
    imag_weights_penalty_enabled_phase3 = True

    imag_w_coeff_phase1 = 1e-3
    imag_w_coeff_phase2 = 1e-3
    imag_w_coeff_phase3 = 1e3

    # -------------------------
    # Imaginary forced shrink (per epoch)
    # -------------------------
    imag_shrink_enabled_phase1 = False
    imag_shrink_enabled_phase2 = False
    imag_shrink_enabled_phase3 = True

    imag_shrink_coeff_phase1 = 0.999
    imag_shrink_coeff_phase2 = 0.999
    imag_shrink_coeff_phase3 = 0.999

    # -------------------------
    # Pruning (per phase)
    # -------------------------
    pruning_enabled_phase1 = False
    pruning_enabled_phase2 = True
    pruning_enabled_phase3 = False

    pruning_start_epoch_phase1 = 5000
    pruning_period_phase1 = 1000
    pruning_threshold_phase1 = 1e-3

    pruning_start_epoch_phase2 = 5000
    pruning_period_phase2 = 2000
    pruning_threshold_phase2 = 1e-2

    pruning_start_epoch_phase3 = 0
    pruning_period_phase3 = 1000
    pruning_threshold_phase3 = 1e-2

    pruning_min_edges_total = 20

    pruning_fraction_phase1 = 0.0
    pruning_fraction_phase2 = 0.2
    pruning_fraction_phase3 = 0.0

    # -------------------------
    # Division normalization
    # -------------------------
    normalize_divisions = True
    normalize_divisions_eps = 1e-12

    # -------------------------
    # Optional clamp (kept from your utils behavior)
    # -------------------------
    clamp_pred = True
    clamp_limit = 1e15

    # Control for trigonometric functions
    use_op_params = True

    op_param_schedules = {
        "sin": {"r": {"start": 0.5, "end": 1.0, "warmup_epochs": 1000}},
        "cos": {"r": {"start": 0.5, "end": 1.0, "warmup_epochs": 1000}},
        "tan": {"r": {"start": 0.5, "end": 1.0, "warmup_epochs": 1000}},
    }
    
    


class CEQLConfig:
    device = ModelTrainingConfig.device
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