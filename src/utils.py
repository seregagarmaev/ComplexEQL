from __future__ import annotations

import os
import random
import time
import math
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.operations import OpParams


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
# Schedules / op-params
# -------------------------
def linear_schedule(epoch: int, start: float, end: float, warmup: int) -> float:
    if warmup <= 0:
        return float(end)
    t = min(max(epoch, 0), warmup) / float(warmup)
    return float(start + (end - start) * t)


def log_schedule(epoch: int, start: float, end: float, warmup: int, eps: float = 1e-12) -> float:
    """
    Log-space interpolation from start -> end over `warmup` epochs.
    """
    start = float(max(start, eps))
    end = float(max(end, eps))
    if warmup <= 0:
        return end
    denom = float(max(warmup - 1, 1))
    t = min(max(epoch, 0), warmup - 1) / denom
    ls = math.log(start)
    le = math.log(end)
    return float(math.exp(ls + (le - ls) * t))


def build_op_params(epoch: int, cfg) -> Optional[OpParams]:
    """
    Kept for backward compatibility; your cycle-based training uses op_params_override.
    """
    if not bool(getattr(cfg, "use_op_params", False)):
        return None

    schedules: Dict[str, Dict[str, Dict[str, Any]]] = getattr(cfg, "op_param_schedules", {}) or {}
    if not schedules:
        return None

    op_params: OpParams = {}

    for opname, params in schedules.items():
        if not isinstance(params, dict):
            continue

        op_kwargs: Dict[str, Any] = {}
        for pname, spec in params.items():
            if not isinstance(spec, dict):
                continue
            start = spec.get("start", None)
            end = spec.get("end", None)
            warmup = spec.get("warmup_epochs", 0)

            if start is None or end is None:
                continue

            op_kwargs[pname] = linear_schedule(epoch, float(start), float(end), int(warmup))

        if op_kwargs:
            op_params[opname] = op_kwargs

    return op_params if op_params else None


def build_trig_op_params(cfg, r_value: float) -> Optional[OpParams]:
    """
    Cycle-driven r for sin/cos/tan. Returns None if cfg.use_op_params is False.
    """
    if not bool(getattr(cfg, "use_op_params", False)):
        return None

    r_value = float(r_value)
    op_params: OpParams = {
        "sin": {"r": r_value},
        "cos": {"r": r_value},
        "tan": {"r": r_value},
    }
    return op_params


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
    cfg,
    *,
    op_params_override: Optional[OpParams] = None,
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
    # imag shrink
    imag_shrink_enabled: bool = False,
    imag_shrink_coeff: float = 1.0,
):
    model.train()
    device = torch.device(device)

    op_params_epoch = op_params_override if (op_params_override is not None) else build_op_params(epoch, cfg)

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

        pred = model(X, op_params=op_params_epoch)

        if clamp_pred:
            pred = torch.complex(
                torch.clamp(pred.real, -clamp_limit, clamp_limit),
                torch.clamp(pred.imag, -clamp_limit, clamp_limit),
            )

        data_loss = loss_fn(pred.real, y)

        # L1L0 sparsity
        if l1l0_enabled and l1l0_coeff > 0.0:
            if l1l0_use_real_only:
                reg_target = model.get_real_weights_list()
            else:
                real_list = model.get_real_weights_list()
                imag_list = model.get_imag_weights_list()
                reg_target = [torch.sqrt(r**2 + i**2) for r, i in zip(real_list, imag_list)]

            real_reg = l1l0_coeff * l1l0_penalty(reg_target, alpha=l1l0_alpha, s=l1l0_s, eps=l1l0_eps)
        else:
            real_reg = torch.tensor(0.0, device=device)

        # imag(weights) penalty
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

    if prune_now and prune_threshold > 0.0:
        before = model.count_active_edges()
        pruned = model.cascade_threshold_prunning(threshold=prune_threshold, eps=1e-12)
        after = model.count_active_edges()
        if pruned > 0:
            print(f"[prune] Epoch {epoch}: pruned {pruned} edges (thr={prune_threshold:g}, active {before}->{after})")

    if normalize_divisions:
        model.normalize_all_divisions_(eps=normalize_divisions_eps)

    if imag_shrink_enabled and imag_shrink_coeff < 1.0:
        model.shrink_imag_weights_(coeff=imag_shrink_coeff)

    denom = max(1, num_samples)
    return (
        total_loss / denom,
        total_data_loss / denom,
        total_reg_loss / denom,
        total_real_reg_loss / denom,
        total_imag_w_reg_loss / denom,
        op_params_epoch,
    )


# -------------------------
# Cycle-based training
# -------------------------
def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    cfg,
    device: torch.device | str,
    scheduler=None,
):
    device = torch.device(device)
    global_epoch = 0
    cycle_idx = 0

    sl0_weights, al_weights = [], []
    data_losses, imag_w_losses = [], []

    opt = optimizer
    sch = scheduler

    def _record(avg_data: float, avg_imag_w_reg: float):
        sl0_weights.append(model.symbolic_layers[0].weights.detach().cpu().clone().numpy().copy())
        al_weights.append(model.assembly_layer.weights.detach().cpu().clone().numpy().copy())
        data_losses.append(avg_data)
        imag_w_losses.append(avg_imag_w_reg)

    def _maybe_print(tag: str, avg_total: float, avg_data: float, avg_sparse: float, avg_imag_w: float, r_val: float):
        if (global_epoch + 1) % cfg.print_every == 0 or global_epoch == 0:
            lr = opt.param_groups[0]["lr"]
            active = model.count_active_edges()
            print(
                f"[{tag} | Epoch {global_epoch+1} | cycle={cycle_idx}] "
                f"lr={lr:.2e}, total={avg_total:.4e}, data={avg_data:.4e}, "
                f"sparsity_reg={avg_sparse:.4e}, imag_w={avg_imag_w:.4e}, "
                f"r={r_val:.4f}, active_edges={active}"
            )

    min_edges = int(cfg.pruning_min_edges_total)
    prune_fraction = float(cfg.pruning_fraction_cycle)

    # -------------------------------------------------
    # Repeat cycles until we reach min_edges
    # -------------------------------------------------
    while model.count_active_edges() > min_edges:
        # ===== Cycle Stage A: ramp r log from start->end, no sparsity
        ramp_epochs = int(cfg.cycle_ramp_epochs)
        r_start = float(cfg.r_start_cycle)
        r_end = float(cfg.r_end_cycle)

        for e in range(ramp_epochs):
            r_val = log_schedule(e, r_start, r_end, ramp_epochs)
            op_params = build_trig_op_params(cfg, r_val)

            avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, _ = train_one_epoch(
                epoch=global_epoch,
                dataloader=dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=opt,
                device=device,
                cfg=cfg,
                op_params_override=op_params,
                l1l0_enabled=False,
                l1l0_use_real_only=cfg.l1l0_on_real_only,
                l1l0_coeff=0.0,
                l1l0_alpha=cfg.l1l0_alpha,
                l1l0_s=cfg.l1l0_s,
                l1l0_eps=cfg.l1l0_eps,
                imag_weights_penalty_enabled=True,
                imag_weights_penalty_coeff=float(cfg.imag_w_coeff_cycle),
                prune_now=False,
                prune_threshold=0.0,
                normalize_divisions=False,
                normalize_divisions_eps=cfg.normalize_divisions_eps,
                clamp_pred=getattr(cfg, "clamp_pred", True),
                clamp_limit=getattr(cfg, "clamp_limit", 1e15),
                imag_shrink_enabled=False,
                imag_shrink_coeff=1.0,
            )

            _maybe_print("RAMP", avg_total, avg_data, avg_sparse, avg_imag_w, r_val)
            _record(avg_data, avg_imag_w)
            global_epoch += 1

        # ===== Cycle Stage B: r=1.0, sparsity ON, division normalization each epoch
        spars_epochs = int(cfg.cycle_sparsity_epochs)
        op_params = build_trig_op_params(cfg, 1.0)

        for e in range(spars_epochs):
            avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, _ = train_one_epoch(
                epoch=global_epoch,
                dataloader=dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=opt,
                device=device,
                cfg=cfg,
                op_params_override=op_params,
                l1l0_enabled=True,
                l1l0_use_real_only=cfg.l1l0_on_real_only,
                l1l0_coeff=float(cfg.l1l0_real_reg_coeff_cycle),
                l1l0_alpha=cfg.l1l0_alpha,
                l1l0_s=cfg.l1l0_s,
                l1l0_eps=cfg.l1l0_eps,
                imag_weights_penalty_enabled=True,
                imag_weights_penalty_coeff=float(cfg.imag_w_coeff_cycle),
                prune_now=False,
                prune_threshold=0.0,
                normalize_divisions=bool(getattr(cfg, "normalize_divisions_during_sparsity", True)),
                normalize_divisions_eps=cfg.normalize_divisions_eps,
                clamp_pred=getattr(cfg, "clamp_pred", True),
                clamp_limit=getattr(cfg, "clamp_limit", 1e15),
                imag_shrink_enabled=False,
                imag_shrink_coeff=1.0,
            )

            _maybe_print("SPARSE", avg_total, avg_data, avg_sparse, avg_imag_w, 1.0)
            _record(avg_data, avg_imag_w)
            global_epoch += 1

        # ===== End-of-cycle pruning: fraction, but cap to keep >= min_edges
        before = model.count_active_edges()
        thr, k_to_prune, active_now = model.pruning_threshold_from_fraction(
            prune_fraction,
            min_edges_total=min_edges,
            eps=1e-12,
        )

        if thr is None or k_to_prune <= 0 or active_now <= min_edges:
            break

        pruned = model.cascade_threshold_prunning(threshold=thr, eps=1e-12)
        after = model.count_active_edges()
        print(f"[PRUNE | cycle={cycle_idx}] pruned={pruned} (fraction={prune_fraction:g}, thr={thr:g}) active {before}->{after}")

        cycle_idx += 1

        if after <= min_edges:
            break

    # -------------------------------------------------
    # Post-cycle finishing:
    #   - sparsity OFF, imag penalty small
    #   - r ramp 0.01->1.0 for 10k
    #   - r=1.0 for last 10k with scheduler ON
    # -------------------------------------------------
    post_ramp_epochs = int(cfg.post_ramp_epochs)
    post_finetune_epochs = int(cfg.post_finetune_epochs)

    r_start_post = float(cfg.r_start_post)
    r_end_post = float(cfg.r_end_post)

    # Post ramp
    for e in range(post_ramp_epochs):
        r_val = log_schedule(e, r_start_post, r_end_post, post_ramp_epochs)
        op_params = build_trig_op_params(cfg, r_val)

        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, _ = train_one_epoch(
            epoch=global_epoch,
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            device=device,
            cfg=cfg,
            op_params_override=op_params,
            l1l0_enabled=False,
            l1l0_use_real_only=cfg.l1l0_on_real_only,
            l1l0_coeff=0.0,
            l1l0_alpha=cfg.l1l0_alpha,
            l1l0_s=cfg.l1l0_s,
            l1l0_eps=cfg.l1l0_eps,
            imag_weights_penalty_enabled=True,
            imag_weights_penalty_coeff=float(cfg.imag_w_coeff_post),
            prune_now=False,
            prune_threshold=0.0,
            normalize_divisions=False,
            normalize_divisions_eps=cfg.normalize_divisions_eps,
            clamp_pred=getattr(cfg, "clamp_pred", True),
            clamp_limit=getattr(cfg, "clamp_limit", 1e15),
            imag_shrink_enabled=False,
            imag_shrink_coeff=1.0,
        )

        _maybe_print("POST_RAMP", avg_total, avg_data, avg_sparse, avg_imag_w, r_val)
        _record(avg_data, avg_imag_w)
        global_epoch += 1

    # Final finetune with r=1.0 and scheduler ON
    op_params = build_trig_op_params(cfg, 1.0)
    for e in range(post_finetune_epochs):
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, _ = train_one_epoch(
            epoch=global_epoch,
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            device=device,
            cfg=cfg,
            op_params_override=op_params,
            l1l0_enabled=False,
            l1l0_use_real_only=cfg.l1l0_on_real_only,
            l1l0_coeff=0.0,
            l1l0_alpha=cfg.l1l0_alpha,
            l1l0_s=cfg.l1l0_s,
            l1l0_eps=cfg.l1l0_eps,
            imag_weights_penalty_enabled=True,
            imag_weights_penalty_coeff=float(cfg.imag_w_coeff_post),
            prune_now=False,
            prune_threshold=0.0,
            normalize_divisions=False,
            normalize_divisions_eps=cfg.normalize_divisions_eps,
            clamp_pred=getattr(cfg, "clamp_pred", True),
            clamp_limit=getattr(cfg, "clamp_limit", 1e15),
            imag_shrink_enabled=False,
            imag_shrink_coeff=1.0,
        )

        if sch is not None:
            try:
                sch.step(avg_data)  # ReduceLROnPlateau
            except TypeError:
                sch.step()

        _maybe_print("POST_FINE", avg_total, avg_data, avg_sparse, avg_imag_w, 1.0)
        _record(avg_data, avg_imag_w)
        global_epoch += 1

    return model, (sl0_weights, al_weights, imag_w_losses, data_losses)
