from __future__ import annotations

import os
import random
import time
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def timeit(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} s")
        return result

    return wrapper


# -------------------------
# L1L0 smooth sparsity penalty
# -------------------------
class L1L0Smooth(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_tensor: torch.Tensor | Iterable[torch.Tensor],
        alpha: float = 1.0,
        s: float = 0.05,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        return l1l0_smooth(input_tensor, alpha=alpha, s=s, eps=eps)


def l1l0_smooth(
    input_tensor: torch.Tensor | Iterable[torch.Tensor],
    alpha: float = 1.0,
    s: float = 0.05,
    eps: float = 1e-12,
) -> torch.Tensor:
    if isinstance(input_tensor, (list, tuple)):
        return sum(l1l0_smooth(t, alpha=alpha, s=s, eps=eps) for t in input_tensor)

    x = input_tensor
    absx = torch.abs(x)

    inner = absx < s
    poly = (x**4) / (-8.0 * s**3) + (x**2) * (3.0 / (4.0 * s)) + 3.0 * s / 8.0
    smooth_abs = torch.where(inner, poly, absx)

    penalty = ((alpha + 1.0) * smooth_abs) / (alpha + smooth_abs + eps)
    return torch.sum(penalty)

# def l1_penalty(
#     input_tensor: torch.Tensor | Iterable[torch.Tensor],
# ) -> torch.Tensor:
#     """
#     Returns sum(|x|) over the given tensor(s).
#     Works for real or complex tensors (torch.abs for complex gives magnitude).
#     """
#     if isinstance(input_tensor, (list, tuple)):
#         return sum(l1_penalty(t) for t in input_tensor)
#     return torch.abs(input_tensor).sum()

# -------------------------
# Train one epoch
# -------------------------
def train_one_epoch(
    epoch: int,
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    # L1L0
    l1l0_enabled: bool,
    l1l0_use_real_only: bool,
    l1l0_coeff: float,
    l1l0_alpha: float,
    l1l0_s: float,
    l1l0_eps: float,
    # imag penalty
    imag_weights_penalty_enabled: bool,
    imag_weights_penalty_coeff: float,
    # pruning
    prune_now: bool,
    prune_threshold: float,
    # division normalization
    normalize_divisions: bool = False,
    normalize_divisions_eps: float = 1e-12,
    # output clamp
    clamp_pred: bool = True,
    clamp_limit: float = 1e15,
    imag_shrink_enabled: bool = False,
    imag_shrink_coeff: float = 1.0,
):
    model.train()
    device = torch.device(device)

    total_data_loss = 0.0
    total_reg_loss = 0.0
    total_real_reg_loss = 0.0
    total_imag_w_reg_loss = 0.0
    total_loss = 0.0
    num_samples = 0

    l1l0_penalty = L1L0Smooth()

    for _, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(X)

        if clamp_pred:
            pred = torch.complex(
                torch.clamp(pred.real, -clamp_limit, clamp_limit),
                torch.clamp(pred.imag, -clamp_limit, clamp_limit),
            )

        data_loss = loss_fn(pred, y)

        # ----------------- L1L0 sparsity -----------------
        if l1l0_enabled and l1l0_coeff > 0.0:
            if l1l0_use_real_only:
                reg_target = model.get_real_weights_list()
            else:
                real_list = model.get_real_weights_list()
                imag_list = model.get_imag_weights_list()
                reg_target = [torch.sqrt(r**2 + i**2) for r, i in zip(real_list, imag_list)]

            real_reg = l1l0_coeff * l1l0_penalty(
                reg_target, alpha=l1l0_alpha, s=l1l0_s, eps=l1l0_eps
            )
        else:
            real_reg = torch.tensor(0.0, device=device)
        # # ----------------- L1 sparsity -----------------
        # if l1l0_enabled and l1l0_coeff > 0.0:
        #     if l1l0_use_real_only:
        #         reg_target = model.get_real_weights_list()  # list of real tensors
        #         real_reg_raw = l1_penalty(reg_target)
        #     else:
        #         # Option A (recommended): L1 on complex magnitudes using the model parameters directly
        #         # (avoid sqrt(r^2+i^2) yourself; abs handles complex)
        #         reg_target = []
        #         for layer in model.symbolic_layers:
        #             w_eff = layer.weights * layer.mask.to(layer.weights.dtype)
        #             reg_target.append(w_eff.reshape(-1))
        #         w_eff = model.assembly_layer.weights * model.assembly_layer.mask.to(model.assembly_layer.weights.dtype)
        #         reg_target.append(w_eff.reshape(-1))
        #         real_reg_raw = l1_penalty(reg_target)

        #         # Option B (your current style): L1 on per-edge magnitudes built from real/imag lists
        #         # real_list = model.get_real_weights_list()
        #         # imag_list = model.get_imag_weights_list()
        #         # reg_target = [torch.sqrt(r**2 + i**2) for r, i in zip(real_list, imag_list)]
        #         # real_reg_raw = l1_penalty(reg_target)

        #     real_reg = l1l0_coeff * real_reg_raw
        # else:
        #     real_reg = torch.tensor(0.0, device=device)

        # ----------------- imag(weights) penalty -----------------
        if imag_weights_penalty_enabled and imag_weights_penalty_coeff > 0.0:
            imag_list = model.get_imag_weights_list()
            imag_w_reg_raw = sum((w**2).sum() for w in imag_list)
            imag_w_reg = imag_weights_penalty_coeff * imag_w_reg_raw
        else:
            imag_w_reg = torch.tensor(0.0, device=device)

        reg_loss = real_reg + imag_w_reg
        loss = data_loss + reg_loss

        loss.backward()
        model.sanitize_gradients(max_grad=1e8)
        optimizer.step()

        bs = X.size(0)
        num_samples += bs
        total_data_loss += data_loss.item() * bs
        total_reg_loss += reg_loss.item() * bs
        total_real_reg_loss += real_reg.item() * bs
        total_imag_w_reg_loss += imag_w_reg.item() * bs
        total_loss += loss.item() * bs

    # pruning by amplitude (|w|) if requested
    if prune_now and prune_threshold > 0.0:
        before = model.count_active_edges()
        pruned = model.cascade_threshold_prunning(threshold=prune_threshold, eps=1e-12)
        after = model.count_active_edges()
        if pruned > 0:
            print(
                f"[prune] Epoch {epoch}: pruned {pruned} edges "
                f"(thr={prune_threshold:g}, active {before}->{after})"
            )

    # normalize division mixing once per epoch
    if normalize_divisions:
        model.normalize_all_divisions_(eps=normalize_divisions_eps)

    # forced shrink of imaginary parts once per epoch
    if imag_shrink_enabled and imag_shrink_coeff < 1.0:
        model.shrink_imag_weights_(coeff=imag_shrink_coeff)
    
    denom = max(1, num_samples)
    return (
        total_loss / denom,
        total_data_loss / denom,
        total_reg_loss / denom,
        total_real_reg_loss / denom,
        total_imag_w_reg_loss / denom,
    )


def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    cfg,
    device: torch.device | str,
    scheduler=None,  # used ONLY in Phase 3 (e.g., ReduceLROnPlateau)
):
    device = torch.device(device)
    global_epoch = 0

    sl0_weights, sl1_weights, al_weights = [], [], []
    data_losses, imag_w_losses = [], []

    opt = optimizer
    sch = scheduler  # do not touch in phases 1/2

    def _run_phase(
        phase_name: str,
        num_epochs: int,
        *,
        # L1L0
        l1l0_enabled: bool,
        l1l0_coeff: float,
        l1l0_use_real_only: bool,
        # imag penalty
        imag_enabled: bool,
        imag_coeff: float,
        # pruning
        pruning_enabled: bool,
        pruning_start_epoch: int,
        pruning_period: int,
        pruning_threshold: float,
        pruning_fraction: float,
        pruning_min_edges_total: int,
        imag_shrink_enabled: bool,
        imag_shrink_coeff: float,
    ):
        nonlocal global_epoch, sl0_weights, sl1_weights, al_weights, data_losses, imag_w_losses
        nonlocal opt, sch

        for e in range(num_epochs):
            prune_now = False
            if pruning_enabled:
                if e >= pruning_start_epoch and (e - pruning_start_epoch) % pruning_period == 0:
                    prune_now = True

            prune_threshold_dynamic = 0.0
            if prune_now:
                thr, k_prune, active = model.pruning_threshold_from_fraction(
                    pruning_fraction,
                    min_edges_total=pruning_min_edges_total,
                    eps=1e-12,
                )
                if thr is None:
                    prune_now = False
                else:
                    prune_threshold_dynamic = thr

            avg_total, avg_data, _avg_reg, avg_real_reg, avg_imag_w_reg = train_one_epoch(
                epoch=global_epoch,
                dataloader=dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=opt,
                device=device,
                # L1L0 (constant alpha from config)
                l1l0_enabled=l1l0_enabled,
                l1l0_use_real_only=l1l0_use_real_only,
                l1l0_coeff=l1l0_coeff,
                l1l0_alpha=cfg.l1l0_alpha,
                l1l0_s=cfg.l1l0_s,
                l1l0_eps=cfg.l1l0_eps,
                # imag penalty (constant per phase)
                imag_weights_penalty_enabled=imag_enabled,
                imag_weights_penalty_coeff=imag_coeff if imag_enabled else 0.0,
                # pruning
                prune_now=prune_now,
                prune_threshold=prune_threshold_dynamic if prune_now else 0.0,
                # division normalization
                normalize_divisions=cfg.normalize_divisions,
                normalize_divisions_eps=cfg.normalize_divisions_eps,
                # clamp
                clamp_pred=getattr(cfg, "clamp_pred", True),
                clamp_limit=getattr(cfg, "clamp_limit", 1e15),
                imag_shrink_enabled=imag_shrink_enabled,
                imag_shrink_coeff=imag_shrink_coeff,
            )

            # Scheduler ONLY in Phase 3
            if (phase_name == "Phase 3") and (sch is not None):
                # ReduceLROnPlateau expects a metric
                sch.step(avg_data)

            if (global_epoch + 1) % cfg.print_every == 0 or global_epoch == 0:
                lr = opt.param_groups[0]["lr"]
                print(
                    f"[{phase_name} | Epoch {global_epoch+1}] "
                    f"total={avg_total:.4e}, data={avg_data:.4e}, "
                    f"sparsity_reg={avg_real_reg:.4e}, imag_w={avg_imag_w_reg:.4e}, "
                    f"alpha={cfg.l1l0_alpha:.3e}, imag_coeff={imag_coeff:.3e}, lr={lr:.2e}"
                )

            global_epoch += 1
            sl0_weights.append(model.symbolic_layers[0].weights.detach().cpu().clone().numpy().copy())
            al_weights.append(model.assembly_layer.weights.detach().cpu().clone().numpy().copy())
            data_losses.append(avg_data)
            imag_w_losses.append(avg_imag_w_reg)

    # --------------------------
    # PHASE 1
    # --------------------------
    _run_phase(
        "Phase 1",
        cfg.phase1_epochs,
        l1l0_enabled=cfg.l1l0_enabled_phase1,
        l1l0_coeff=cfg.l1l0_real_reg_coeff_phase1,
        l1l0_use_real_only=cfg.l1l0_on_real_only,
        imag_enabled=cfg.imag_weights_penalty_enabled_phase1,
        imag_coeff=cfg.imag_w_coeff_phase1,
        pruning_enabled=cfg.pruning_enabled_phase1,
        pruning_start_epoch=cfg.pruning_start_epoch_phase1,
        pruning_period=cfg.pruning_period_phase1,
        pruning_threshold=cfg.pruning_threshold_phase1,
        pruning_fraction=cfg.pruning_fraction_phase1,
        pruning_min_edges_total=cfg.pruning_min_edges_total,
        imag_shrink_enabled=cfg.imag_shrink_enabled_phase1,
        imag_shrink_coeff=cfg.imag_shrink_coeff_phase1,
    )

    # --------------------------
    # PHASE 2
    # --------------------------
    _run_phase(
        "Phase 2",
        cfg.phase2_epochs,
        l1l0_enabled=cfg.l1l0_enabled_phase2,
        l1l0_coeff=cfg.l1l0_real_reg_coeff_phase2,
        l1l0_use_real_only=cfg.l1l0_on_real_only,
        imag_enabled=cfg.imag_weights_penalty_enabled_phase2,
        imag_coeff=cfg.imag_w_coeff_phase2,
        pruning_enabled=cfg.pruning_enabled_phase2,
        pruning_start_epoch=cfg.pruning_start_epoch_phase2,
        pruning_period=cfg.pruning_period_phase2,
        pruning_threshold=cfg.pruning_threshold_phase2,
        pruning_fraction=cfg.pruning_fraction_phase2,
        pruning_min_edges_total=cfg.pruning_min_edges_total,
        imag_shrink_enabled=cfg.imag_shrink_enabled_phase2,
        imag_shrink_coeff=cfg.imag_shrink_coeff_phase2,
    )

    # --------------------------
    # PHASE 3
    # --------------------------
    _run_phase(
        "Phase 3",
        cfg.phase3_epochs,
        l1l0_enabled=cfg.l1l0_enabled_phase3,
        l1l0_coeff=cfg.l1l0_real_reg_coeff_phase3,
        l1l0_use_real_only=cfg.l1l0_on_real_only,
        imag_enabled=cfg.imag_weights_penalty_enabled_phase3,
        imag_coeff=cfg.imag_w_coeff_phase3,
        pruning_enabled=cfg.pruning_enabled_phase3,
        pruning_start_epoch=cfg.pruning_start_epoch_phase3,
        pruning_period=cfg.pruning_period_phase3,
        pruning_threshold=cfg.pruning_threshold_phase3,
        pruning_fraction=cfg.pruning_fraction_phase3,
        pruning_min_edges_total=cfg.pruning_min_edges_total,
        imag_shrink_enabled=cfg.imag_shrink_enabled_phase3,
        imag_shrink_coeff=cfg.imag_shrink_coeff_phase3,
    )

    return model, (sl0_weights, sl1_weights, al_weights, imag_w_losses, data_losses)
