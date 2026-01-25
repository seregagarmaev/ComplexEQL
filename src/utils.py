from __future__ import annotations

import os
import random
import time
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


def build_op_params(epoch: int, cfg) -> Optional[OpParams]:
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


# -------------------------
# L1 sparsity penalty
# -------------------------
def l1_penalty(
    tensors: torch.Tensor | Iterable[torch.Tensor],
    *,
    use_real_only: bool,
    eps: float = 1e-12,
) -> torch.Tensor:
    if isinstance(tensors, (list, tuple)):
        return sum(l1_penalty(t, use_real_only=use_real_only, eps=eps) for t in tensors)

    w = tensors

    if torch.is_complex(w):
        if use_real_only:
            return w.real.abs().sum()
        return torch.sqrt(w.real * w.real + w.imag * w.imag + eps).sum()

    return w.abs().sum()

def l1_reg_fast(model: torch.nn.Module, *, use_real_only: bool, eps: float) -> torch.Tensor:
    acc: torch.Tensor | None = None

    for layer in model.symbolic_layers:
        w = layer.weights * layer.mask.to(layer.weights.dtype)
        term = w.real.abs().sum() if use_real_only else torch.sqrt(w.real * w.real + w.imag * w.imag + eps).sum()
        acc = term if acc is None else (acc + term)

    wA = model.assembly_layer.weights * model.assembly_layer.mask.to(model.assembly_layer.weights.dtype)
    termA = wA.real.abs().sum() if use_real_only else torch.sqrt(wA.real * wA.real + wA.imag * wA.imag + eps).sum()
    return termA if acc is None else (acc + termA)


def imag_w_l2_reg_fast(model: torch.nn.Module) -> torch.Tensor:
    acc: torch.Tensor | None = None

    for layer in model.symbolic_layers:
        w = layer.weights * layer.mask.to(layer.weights.dtype)
        term = (w.imag * w.imag).sum()
        acc = term if acc is None else (acc + term)

    wA = model.assembly_layer.weights * model.assembly_layer.mask.to(model.assembly_layer.weights.dtype)
    termA = (wA.imag * wA.imag).sum()
    return termA if acc is None else (acc + termA)


def finite_mask_pred_and_target(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    pred_abs_max: float,
) -> torch.Tensor:
    # Finite check
    if torch.is_complex(pred):
        pred_finite = torch.isfinite(pred.real) & torch.isfinite(pred.imag)
        # you train on pred.real, so bound that
        pred_mag = pred.real.abs()
    else:
        pred_finite = torch.isfinite(pred)
        pred_mag = pred.abs()

    # Bound check (elementwise)
    pred_ok = pred_finite & (pred_mag <= float(pred_abs_max))

    # Reduce pred over non-batch dims to per-sample mask (B,)
    while pred_ok.dim() > 1:
        pred_ok = pred_ok.all(dim=-1)

    # Target finite check
    y_ok = torch.isfinite(y)
    while y_ok.dim() > 1:
        y_ok = y_ok.all(dim=-1)

    return pred_ok & y_ok


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
    # pruning (kept for compatibility; will remove later)
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

    for _, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred = model(X, op_params=op_params_epoch)

        pred_abs_max = float(getattr(cfg, "pred_abs_max", 1e18))
        valid = finite_mask_pred_and_target(pred, y, pred_abs_max=pred_abs_max)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            # skip this batch
            continue

        data_loss = loss_fn(pred.real[valid], y[valid])

        if l1_enabled and l1_coeff > 0.0:
            real_reg = float(l1_coeff) * l1_reg_fast(
                model,
                use_real_only=bool(l1_use_real_only),
                eps=float(l1_eps),
            )
        else:
            real_reg = pred.real.new_tensor(0.0)

        if imag_weights_penalty_enabled and imag_weights_penalty_coeff > 0.0:
            imag_w_reg = float(imag_weights_penalty_coeff) * imag_w_l2_reg_fast(model)
        else:
            imag_w_reg = pred.real.new_tensor(0.0)

        reg_loss = real_reg + imag_w_reg
        loss = data_loss + reg_loss

        loss.backward()
        model.sanitize_gradients(max_grad=1e8)
        optimizer.step()

        bs = n_valid
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
            print(
                f"[prune] Epoch {epoch}: pruned {pruned} edges "
                f"(thr={prune_threshold:g}, active {before}->{after})"
            )

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
# Phase-based training
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

    data_losses: list[float] = []
    imag_w_losses: list[float] = []

    opt = optimizer
    sch = scheduler

    def _rebuild_optimizer_like(old_opt: torch.optim.Optimizer, new_model: torch.nn.Module) -> torch.optim.Optimizer:
        if isinstance(old_opt, torch.optim.Adam):
            return torch.optim.Adam(
                new_model.parameters(),
                lr=old_opt.param_groups[0]["lr"],
                betas=old_opt.param_groups[0].get("betas", (0.9, 0.999)),
                eps=old_opt.param_groups[0].get("eps", 1e-8),
                weight_decay=old_opt.param_groups[0].get("weight_decay", 0.0),
                amsgrad=old_opt.param_groups[0].get("amsgrad", False),
            )
        if isinstance(old_opt, torch.optim.AdamW):
            return torch.optim.AdamW(
                new_model.parameters(),
                lr=old_opt.param_groups[0]["lr"],
                betas=old_opt.param_groups[0].get("betas", (0.9, 0.999)),
                eps=old_opt.param_groups[0].get("eps", 1e-8),
                weight_decay=old_opt.param_groups[0].get("weight_decay", 0.01),
                amsgrad=old_opt.param_groups[0].get("amsgrad", False),
            )
        raise ValueError(f"Unsupported optimizer type for rebuild: {type(old_opt)}")

    def _rebuild_scheduler_like(old_sch, new_opt: torch.optim.Optimizer, cfg):
        if old_sch is None:
            return None
        if isinstance(old_sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(new_opt, **getattr(cfg, "schedulerparams", {}))
        raise ValueError(f"Unsupported scheduler type for rebuild: {type(old_sch)}")

    def _record(avg_data: float, avg_imag_w_reg: float):
        data_losses.append(float(avg_data))
        imag_w_losses.append(float(avg_imag_w_reg))

    def _maybe_print(tag: str, avg_total: float, avg_data: float, avg_sparse: float, avg_imag_w: float):
        if (global_epoch + 1) % cfg.print_every == 0 or global_epoch == 0:
            lr = opt.param_groups[0]["lr"]
            active = model.count_active_edges()
            print(
                f"[{tag} | Epoch {global_epoch+1}] "
                f"lr={lr:.2e}, total={avg_total:.4e}, data={avg_data:.4e}, "
                f"sparsity_reg={avg_sparse:.4e}, imag_w={avg_imag_w:.4e}, "
                f"active_edges={active}"
            )

    thr_min = float(getattr(cfg, "pruning_threshold_min", 0.0))
    thr_max = float(getattr(cfg, "pruning_threshold_max", float("inf")))
    min_edges_layer = int(getattr(cfg, "pruning_min_edges_per_layer", 0))

    def _do_prune_and_rebuild(tag: str, prune_fraction: float):
        nonlocal model, opt, sch

        before_total = model.count_active_edges()
        before_sym, before_asm = model.count_active_edges_per_layer()

        per_sym, asm = model.pruning_thresholds_from_fraction_per_layer(
            prune_fraction,
            min_edges_per_layer=min_edges_layer,
            eps=1e-12,
        )

        pruned_total = 0

        for li, (thr, k_to_prune, active_now) in enumerate(per_sym):
            if thr is None or k_to_prune <= 0 or active_now <= min_edges_layer:
                continue

            thr = float(thr)
            if thr < thr_min:
                thr = thr_min
            if thr > thr_max:
                thr = thr_max

            pruned_here = model.symbolic_layers[li].prune_by_threshold(thr)
            pruned_total += pruned_here
            if pruned_here > 0:
                after_li = int((model.symbolic_layers[li].mask > 0.5).sum().item())
                print(
                    f"[PRUNE_SYM | {tag} | layer={li}] "
                    f"pruned={pruned_here} (fraction={prune_fraction:g}, thr={thr:g}) "
                    f"active {before_sym[li]}->{after_li}"
                )

        thrA, kA, activeA = asm
        if thrA is not None and kA > 0 and activeA > min_edges_layer:
            thrA = float(thrA)
            if thrA < thr_min:
                thrA = thr_min
            if thrA > thr_max:
                thrA = thr_max

            prunedA = model.assembly_layer.prune_by_threshold(thrA)
            pruned_total += prunedA
            if prunedA > 0:
                afterA = int((model.assembly_layer.mask > 0.5).sum().item())
                print(
                    f"[PRUNE_ASM | {tag}] "
                    f"pruned={prunedA} (fraction={prune_fraction:g}, thr={thrA:g}) "
                    f"active {before_asm}->{afterA}"
                )

        dropped = model.drop_div_ops_with_pruned_denominator_()
        if dropped > 0:
            print(f"[DROP_DIV | {tag}] pruned_downstream_edges={dropped}")
        
        cleaned = model.cascade_cleanup_disconnected_()
        if cleaned > 0:
            print(f"[CLEAN | {tag}] cleaned_disconnected={cleaned}")

        before_rebuild_total = model.count_active_edges()
        model = model.rebuild_from_pruned().to(device)
        opt = _rebuild_optimizer_like(opt, model)
        sch = _rebuild_scheduler_like(sch, opt, cfg)

        after_total = model.count_active_edges()
        if pruned_total > 0:
            print(f"[PRUNE | {tag}] pruned_total={pruned_total} active {before_total}->{after_total}")
        else:
            if after_total != before_rebuild_total:
                print(f"[REBUILD | {tag}] active {before_rebuild_total}->{after_total}")

    # -------------------------
    # Phase 1
    # -------------------------
    phase1_epochs = int(getattr(cfg, "phase1_epochs", 0))
    for _ in range(phase1_epochs):
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, _ = train_one_epoch(
            epoch=global_epoch,
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            device=device,
            cfg=cfg,
            op_params_override=None,
            l1_enabled=True,
            l1_use_real_only=cfg.l1_on_real_only,
            l1_coeff=float(getattr(cfg, "l1_reg_coeff_phase1", 0.0)),
            l1_eps=float(getattr(cfg, "l1_eps", 1e-12)),
            imag_weights_penalty_enabled=True,
            imag_weights_penalty_coeff=float(getattr(cfg, "imag_w_coeff_phase1", 0.0)),
            prune_now=False,
            prune_threshold=0.0,
            normalize_divisions=False,
            normalize_divisions_eps=cfg.normalize_divisions_eps,
            clamp_pred=getattr(cfg, "clamp_pred", False),
            clamp_limit=getattr(cfg, "clamp_limit", 1e15),
            imag_shrink_enabled=False,
            imag_shrink_coeff=1.0,
        )
        _maybe_print("PHASE1", avg_total, avg_data, avg_sparse, avg_imag_w)
        _record(avg_data, avg_imag_w)
        global_epoch += 1
    
    # -------------------------
    # END PHASE 1: normalize -> prune -> cleanup -> rebuild -> normalize
    # -------------------------
    phase1_prune_enabled = bool(getattr(cfg, "phase1_prune_enabled", True))
    phase1_prune_thr = float(getattr(cfg, "phase1_prune_threshold", 0.0))

    if phase1_prune_enabled and phase1_prune_thr > 0.0:
        tag = "PHASE1_END"

        # (1) normalize divisions first
        model.normalize_all_divisions_(eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)))

        before = model.count_active_edges()

        # (2) prune by threshold (your SymbolicLayer.prune_by_threshold must include the div-safe coupling)
        pruned = model.prune_by_threshold(phase1_prune_thr)

        dropped = model.drop_div_ops_with_pruned_denominator_()
        cleaned = model.cascade_cleanup_disconnected_()


        # (4) rebuild compact model + rebuild optimizer/scheduler
        before_rebuild = model.count_active_edges()
        model = model.rebuild_from_pruned().to(device)
        opt = _rebuild_optimizer_like(opt, model)
        sch = _rebuild_scheduler_like(sch, opt, cfg)

        after = model.count_active_edges()

        # (5) normalize again after structure changed
        model.normalize_all_divisions_(eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)))

        print(
            f"[{tag}] thr={phase1_prune_thr:g} pruned={pruned} cleaned={cleaned} "
            f"active {before}->{after} (pre_rebuild={before_rebuild})"
        )

    # -------------------------
    # Phase 2
    # -------------------------
    phase2_epochs = int(getattr(cfg, "phase2_epochs", 0))
    prune_every = int(getattr(cfg, "prune_every_epochs", 0))
    prune_fraction = float(getattr(cfg, "pruning_fraction_phase2", 0.0))
    normalize_divs = bool(getattr(cfg, "normalize_divisions_during_phase2", True))

    for e in range(phase2_epochs):
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, _ = train_one_epoch(
            epoch=global_epoch,
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            device=device,
            cfg=cfg,
            op_params_override=None,
            l1_enabled=True,
            l1_use_real_only=cfg.l1_on_real_only,
            l1_coeff=float(getattr(cfg, "l1_reg_coeff_phase2", 0.0)),
            l1_eps=float(getattr(cfg, "l1_eps", 1e-12)),
            imag_weights_penalty_enabled=True,
            imag_weights_penalty_coeff=float(getattr(cfg, "imag_w_coeff_phase2", 0.0)),
            prune_now=False,
            prune_threshold=0.0,
            normalize_divisions=normalize_divs,
            normalize_divisions_eps=cfg.normalize_divisions_eps,
            clamp_pred=getattr(cfg, "clamp_pred", True),
            clamp_limit=getattr(cfg, "clamp_limit", 1e15),
            imag_shrink_enabled=False,
            imag_shrink_coeff=1.0,
        )
        _maybe_print("PHASE2", avg_total, avg_data, avg_sparse, avg_imag_w)
        _record(avg_data, avg_imag_w)
        global_epoch += 1

        prune_warmup = int(getattr(cfg, "phase2_prune_warmup_epochs", 0))
        if (
            prune_every > 0
            and prune_fraction > 0.0
            and (e + 1) >= prune_warmup
            and ((e + 1 - prune_warmup) % prune_every == 0)
        ):
            _do_prune_and_rebuild(tag=f"PHASE2_E{e+1}", prune_fraction=prune_fraction)

    # -------------------------
    # Phase 3
    # -------------------------
    phase3_epochs = int(getattr(cfg, "phase3_epochs", 0))
    for _ in range(phase3_epochs):
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, _ = train_one_epoch(
            epoch=global_epoch,
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            device=device,
            cfg=cfg,
            op_params_override=None,
            l1_enabled=False,
            l1_use_real_only=cfg.l1_on_real_only,
            l1_coeff=0.0,
            l1_eps=float(getattr(cfg, "l1_eps", 1e-12)),
            imag_weights_penalty_enabled=True,
            imag_weights_penalty_coeff=float(getattr(cfg, "imag_w_coeff_phase3", 0.0)),
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
                sch.step(avg_data)
            except TypeError:
                sch.step()

        _maybe_print("PHASE3", avg_total, avg_data, avg_sparse, avg_imag_w)
        _record(avg_data, avg_imag_w)
        global_epoch += 1

    return model, (imag_w_losses, data_losses)