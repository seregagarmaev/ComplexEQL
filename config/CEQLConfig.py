import sympy as sp


class ModelTrainingConfig:
    device = "cuda:2"
    loss_function = "MSELoss"
    train_batch_size = 2**14

    # -------------------------
    # Optimization
    # -------------------------
    lr = 1e-3
    scheduler = "ReduceLROnPlateau"
    schedulerparams = dict(mode="min", patience=100, factor=0.1, min_lr=1e-4)
    scheduler_warmup_phase2 = 500

    # -------------------------
    # Phase control
    # -------------------------
    phase1_epochs = 10000
    phase2_epochs = 10000
    phase3_epochs = 5000
    print_every = 100

    # ==========================================================
    # Imaginary-weights anneal + plateau-reset (shared for phases 1/2)
    # ==========================================================
    imag_weights_penalty_enabled_phase1 = True
    imag_weights_penalty_enabled_phase2 = True
    imag_weights_penalty_enabled_phase3 = True

    imag_anneal_epochs = 1000
    imag_w_coeff_start = 1e-6
    imag_w_coeff_end = 1e3
    imag_anneal_mode = "exp"  # "linear" or "exp"

    # plateau logic: after anneal in each cycle, monitor data loss
    imag_plateau_patience = 500
    imag_plateau_rel_tol = 1e-4
    imag_plateau_check_after_anneal = True  # start plateau counting only after anneal is finished in the cycle
    imag_plateau_min_epoch_in_phase = 0     # additional warmup in each phase before plateau logic starts

    # when plateau triggers: reinit Im(weights) and restart anneal cycle
    imag_reinit_scale = 0.1  # Im ~ U(-scale/2, scale/2) for active weights

    # ==========================================================
    # Sparsification (Phase 2 only): L1L0 on REAL(weights) only
    # ==========================================================
    l1l0_enabled_phase1 = False
    l1l0_enabled_phase2 = True
    l1l0_enabled_phase3 = False

    l1l0_on_real_only = True
    l1l0_real_reg_coeff_phase2 = 1e-2

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

    pruning_start_epoch = 2000
    pruning_period = 500
    pruning_threshold = 1e-2

    # ==========================================================
    # Phase 3: keep imag penalty "big" (fixed), no anneal/reset, no sparsity/pruning
    # ==========================================================
    imag_w_coeff_phase3 = imag_w_coeff_end  # fixed coefficient in phase 3
    
    


class NomtoConfig:
    device = ModelTrainingConfig.device
    no_params_list = [
        [
            {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
            {'library_function': 'const',  'library_function_type': 'unary',  'in_channels': 1},
            # {'library_function': 'square', 'library_function_type': 'unary',  'in_channels': 1},
            # {'library_function': 'cube',   'library_function_type': 'unary',  'in_channels': 1},
            # {'library_function': 'sqrt',   'library_function_type': 'unary',  'in_channels': 1},
            # {'library_function': 'log',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-8},
            # {'library_function': 'mul',    'library_function_type': 'binary', 'in_channels': 2},
            {'library_function': 'div',    'library_function_type': 'binary', 'in_channels': 2, 'stair_step_size': 1e-8},
            {'library_function': 'div',    'library_function_type': 'binary', 'in_channels': 2, 'stair_step_size': 1e-8},
            {'library_function': 'div',    'library_function_type': 'binary', 'in_channels': 2, 'stair_step_size': 1e-8},
            # {'library_function': 'pow',    'library_function_type': 'binary', 'in_channels': 2, 'stair_step_size': 5.0},
        ],
        # [
        #     {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
        #     {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
        #     {'library_function': 'id',     'library_function_type': 'unary',  'in_channels': 1},
        #     {'library_function': 'const',  'library_function_type': 'unary',  'in_channels': 1},
        #     {'library_function': 'square', 'library_function_type': 'unary',  'in_channels': 1},
        #     {'library_function': 'cube',   'library_function_type': 'unary',  'in_channels': 1},
        #     {'library_function': 'sqrt',   'library_function_type': 'unary',  'in_channels': 1},
        #     {'library_function': 'log',    'library_function_type': 'unary',  'in_channels': 1, 'stair_step_size': 1e-8},
        #     {'library_function': 'mul',    'library_function_type': 'binary', 'in_channels': 2},
        #     {'library_function': 'div',    'library_function_type': 'binary', 'in_channels': 2, 'stair_step_size': 1e-8},
        #     {'library_function': 'pow',    'library_function_type': 'binary', 'in_channels': 2, 'stair_step_size': 5.0},
        # ],
    ]

    n_input_fields = 2
    n_symbolic_layers = len(no_params_list)

    functions_dict = {
        'x':      sp.Symbol('x'),
        'id':     lambda x: x,
        'const':  lambda x: sp.Integer(1),
        'square': lambda x: x**2,
        'cube':   lambda x: x**3,
        'sqrt':   lambda x: sp.sqrt(x),
        'sin':    lambda x: sp.sin(x),
        'log':    lambda x: sp.log(x),
        'mul':    lambda a, b: a * b,
        'div':    lambda a, b: a / b,
        'pow':    lambda a, b: a**b,
        
    }