import sympy as sp



class CEQLModelTrainingConfig:
    device = "cpu"
    loss_function = "MSELoss"
    train_batch_size = 10000 #2**14
    num_workers = 10

    lr = 1e-3
    scheduler = "ReduceLROnPlateau"
    schedulerparams = dict(mode="min", patience=1000, factor=0.1, min_lr=1e-8)

    print_every = 1000

    l1_on_real_only = False
    l1_eps = 1e-12

    normalize_divisions_eps = 1e-12

    clamp_pred = True
    clamp_limit = 1e6

    use_op_params = False
    op_param_schedules = {}

    # =========================================================
    # Phase-based training strategy
    # =========================================================

    # Phase 1: data + small L1(|w|) + small imag penalty
    phase1_epochs = 20000
    l1_reg_coeff_phase1 = 0 #1e-7
    imag_w_coeff_phase1 = 0 #1e-7

    # Phase 2: higher sparsity + periodic pruning (pruning logic stays in utils.train)
    phase2_epochs = 0#40000
    l1_reg_coeff_phase2 = 1e-7
    imag_w_coeff_phase2 = 1e-7

    pruning_fraction_phase2 = 0.2
    pruning_threshold_min = 1e-3
    pruning_threshold_max = 10.0
    pruning_min_edges_per_layer = 10
    phase2_prune_warmup_epochs = 0
    prune_every_epochs = 5000

    normalize_divisions_during_phase2 = True

    # Phase 3: sparsity OFF, imag penalty bigger, data fit
    phase3_epochs = 0 # 10000
    l1_reg_coeff_phase3 = 0.0
    imag_w_coeff_phase3 = 1e-2


class CEQLConfig:
    device = CEQLModelTrainingConfig.device

    no_params_list = [
        # [
        #     {"op": "id",     "type": "unary"},
        #     {"op": "id",     "type": "unary"},
        #     {"op": "const",  "type": "unary"},
        #     {"op": "square", "type": "unary"},
        #     {"op": "sqrt",   "type": "unary"},
        #     # {"op": "log",    "type": "unary"},
        #     # {"op": "exp",    "type": "unary"},
        #     {"op": "mul",    "type": "binary"},
        # ],
        [
            {"op": "id",     "type": "unary"},
            {"op": "const",  "type": "unary"},
            # {"op": "div",    "type": "binary"},
            {"op": "resonator",    "type": "binary"},
            {"op": "resonator",    "type": "binary"},
            {"op": "resonator",    "type": "binary"},
        ],
    ]

    n_input_fields = 9
    n_symbolic_layers = len(no_params_list)

    functions_dict = {
        "x":      sp.Symbol("x"),
        "id":     lambda x: x,
        "const":  lambda x: sp.Integer(1),
        "square": lambda x: x**2,
        "cube":   lambda x: x**3,
        "sqrt":   lambda x: sp.sqrt(x),
        "exp":    lambda x: sp.exp(x),
        "log":    lambda x: sp.log(x),
        "mul":    lambda a, b: a * b,
        "div":    lambda a, b: a / b,
        "resonator": lambda a, b: sp.abs(a / (b**2)),
    }


CEQL_TRAIN = CEQLModelTrainingConfig()
CEQL = CEQLConfig()