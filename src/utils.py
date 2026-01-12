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
# L1 sparsity penalty
# -------------------------
def l1_penalty(
    tensors: torch.Tensor | Iterable[torch.Tensor],
    *,
    use_real_only: bool,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    L1 penalty for real or complex weights.

    If use_real_only:
        sum(|Re(w)|)
    else:
        sum(|w|) where |w| = sqrt(Re(w)^2 + Im(w)^2 + eps) for complex weights.
    """
    if isinstance(tensors, (list, tuple)):
        return sum(l1_penalty(t, use_real_only=use_real_only, eps=eps) for t in tensors)

    w = tensors

    if torch.is_complex(w):
        if use_real_only:
            return w.real.abs().sum()
        return torch.sqrt(w.real * w.real + w.imag * w.imag + eps).sum()

    return w.abs().sum()


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
    # L1
    l1_enabled: bool,
    l1_use_real_only: bool,
    l1_coeff: float,
    l1_eps: float,
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
    total_real_reg_loss = 0.0  # now: L1 reg (kept name for backward compatibility)
    total_imag_w_reg_loss = 0.0
    total_loss = 0.0
    num_samples = 0

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

        # L1 sparsity
        if l1_enabled and l1_coeff > 0.0:
            if l1_use_real_only:
                # penalize only real-part weights (they are real tensors already)
                reg_target = model.get_real_weights_list()
                real_reg = l1_coeff * l1_penalty(reg_target, use_real_only=False, eps=l1_eps)
            else:
                # penalize complex magnitude: |w| = sqrt(Re^2 + Im^2)
                real_list = model.get_real_weights_list()
                imag_list = model.get_imag_weights_list()
                reg_target = [torch.complex(r, i) for r, i in zip(real_list, imag_list)]
                real_reg = l1_coeff * l1_penalty(reg_target, use_real_only=False, eps=l1_eps)
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

    min_edges_layer = int(cfg.pruning_min_edges_per_layer)
    prune_fraction = float(cfg.pruning_fraction_cycle)

    # -------------------------------------------------
    # Repeat cycles until we reach min_edges
    # -------------------------------------------------
    while True:
        sym_now, asm_now = model.count_active_edges_per_layer()
        if not (any(n > min_edges_layer for n in sym_now) or (asm_now > min_edges_layer)):
            break
        # ===== Cycle Stage A: ramp r log from start->end, no sparsity
        ramp_epochs = int(cfg.cycle_ramp_epochs)
        r_start = float(cfg.r_start_cycle)
        r_end = float(cfg.r_end_cycle)

        for e in range(ramp_epochs):
            # r_val = log_schedule(e, r_start, r_end, ramp_epochs)
            r_val = linear_schedule(e, r_start, r_end, ramp_epochs)
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
                l1_enabled=False,
                l1_use_real_only=cfg.l1_on_real_only,
                l1_coeff=0.0,
                l1_eps=float(getattr(cfg, "l1_eps", 1e-12)),
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
                l1_enabled=True,
                l1_use_real_only=cfg.l1_on_real_only,
                l1_coeff=float(cfg.l1_reg_coeff_cycle),
                l1_eps=float(getattr(cfg, "l1_eps", 1e-12)),
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

        # ===== End-of-cycle pruning: PER-LAYER thresholds, PER-LAYER min edges
        min_edges_layer = int(cfg.pruning_min_edges_per_layer)

        before_total = model.count_active_edges()
        before_sym, before_asm = model.count_active_edges_per_layer()

        per_sym, asm = model.pruning_thresholds_from_fraction_per_layer(
            prune_fraction,
            min_edges_per_layer=min_edges_layer,
            eps=1e-12,
        )

        pruned_total = 0

        # prune each symbolic layer independently
        for li, (thr, k_to_prune, active_now) in enumerate(per_sym):
            if thr is None or k_to_prune <= 0 or active_now <= min_edges_layer:
                continue
            pruned_here = model.symbolic_layers[li].prune_by_threshold(thr)
            pruned_total += pruned_here
            if pruned_here > 0:
                print(
                    f"[PRUNE_SYM | cycle={cycle_idx} | layer={li}] "
                    f"pruned={pruned_here} (fraction={prune_fraction:g}, thr={thr:g}) "
                    f"active {before_sym[li]}->{int((model.symbolic_layers[li].mask > 0.5).sum().item())}"
                )

        # prune assembly independently
        thrA, kA, activeA = asm
        if thrA is not None and kA > 0 and activeA > min_edges_layer:
            prunedA = model.assembly_layer.prune_by_threshold(thrA)
            pruned_total += prunedA
            if prunedA > 0:
                afterA = int((model.assembly_layer.mask > 0.5).sum().item())
                print(
                    f"[PRUNE_ASM | cycle={cycle_idx}] "
                    f"pruned={prunedA} (fraction={prune_fraction:g}, thr={thrA:g}) "
                    f"active {before_asm}->{afterA}"
                )
        
        # After per-layer pruning, remove levitating/disconnected upstream structure
        cleaned = model.cascade_cleanup_disconnected_()
        if cleaned > 0:
            print(f"[CLEAN | cycle={cycle_idx}] cleaned_disconnected={cleaned}")

        after_total = model.count_active_edges()
        if pruned_total > 0:
            print(f"[PRUNE | cycle={cycle_idx}] pruned_total={pruned_total} active {before_total}->{after_total}")

        cycle_idx += 1

        # Stop cycling if no layer can prune any further (given min per-layer)
        sym_now, asm_now = model.count_active_edges_per_layer()
        can_prune_more = any(n > min_edges_layer for n in sym_now) or (asm_now > min_edges_layer)
        if (pruned_total == 0) or (not can_prune_more):
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
        # r_val = log_schedule(e, r_start_post, r_end_post, post_ramp_epochs)
        r_val = linear_schedule(e, r_start_post, r_end_post, post_ramp_epochs)
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
            l1_enabled=False,
            l1_use_real_only=cfg.l1_on_real_only,
            l1_coeff=0.0,
            l1_eps=float(getattr(cfg, "l1_eps", 1e-12)),
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
            l1_enabled=False,
            l1_use_real_only=cfg.l1_on_real_only,
            l1_coeff=0.0,
            l1_eps=float(getattr(cfg, "l1_eps", 1e-12)),
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
