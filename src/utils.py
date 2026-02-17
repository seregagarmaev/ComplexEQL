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


def nonneg_real_input_penalty_fast(
    model: torch.nn.Module,
    *,
    eps: float = 0.0,
    squared: bool = True,
) -> torch.Tensor:
    infos = model.get_unary_penalty_inputs(detach=False)  # or get_angle_penalty_inputs if you didn't rename
    if len(infos) == 0:
        p = next(model.parameters())
        base = p.real if torch.is_complex(p) else p
        return base.new_tensor(0.0)

    acc = None
    cnt = 0

    for d in infos:
        z = d["z"]  # complex
        r = z.real  # real part
        mask = torch.isfinite(r)
        if eps > 0.0:
            # optional margin: allow small negatives within [-eps, 0]
            mask = mask & (r < 1e30)  # no-op but keeps pattern similar

        if not mask.any():
            continue

        # penalty only when r < 0
        neg = torch.relu(-r[mask])  # = max(0, -Re(z))
        term = (neg * neg).mean() if squared else neg.mean()

        acc = term if acc is None else (acc + term)
        cnt += 1

    if cnt == 0:
        p = next(model.parameters())
        base = p.real if torch.is_complex(p) else p
        return base.new_tensor(0.0)

    return acc / float(cnt)


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


@torch.no_grad()
def _apply_prune_mask_entries_(
    mask: torch.Tensor,
    weights: torch.Tensor,
    idx_flat: torch.Tensor,
) -> None:
    # idx_flat: indices into mask.view(-1) / weights.view(-1)
    m = mask.view(-1)
    w = weights.view(-1)
    m[idx_flat] = 0.0
    # zero complex entries fully
    if torch.is_complex(w):
        w[idx_flat] = torch.complex(w.real[idx_flat].new_zeros(idx_flat.shape), w.real[idx_flat].new_zeros(idx_flat.shape))
    else:
        w[idx_flat] = w[idx_flat].new_zeros(idx_flat.shape)


def importance_prune_on_batch(
    *,
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn,
    cfg,
    frac: float,
    device: torch.device | str,
    op_params=None,
    pred_abs_max: float | None = None,
    min_edges_total: int = 0,
) -> int:
    """
    Prune `frac` of currently-active edges with smallest Taylor importance |w*g|.
    Uses ONLY batch data loss (pred.real vs y) for importance.
    Returns number of edges pruned (mask entries zeroed).
    """
    device = torch.device(device)
    frac = float(frac)
    if frac <= 0.0:
        return 0

    model.train()  # important: grads through same graph as training
    model.zero_grad(set_to_none=True)

    X = X.to(device)
    y = y.to(device)

    # forward
    model.clear_unary_input_cache_()
    pred = model(X, op_params=op_params)

    if pred_abs_max is None:
        pred_abs_max = float(getattr(cfg, "pred_abs_max", 1e20))

    valid = finite_mask_pred_and_target(pred, y, pred_abs_max=float(pred_abs_max))
    if int(valid.sum().item()) == 0:
        return 0

    data_loss = loss_fn(pred.real[valid], y[valid])
    data_loss.backward()

    # collect active params and scores
    scores = []
    refs = []  # (layer_obj, flat_index_tensor)
    n_active = 0

    def _score_tensor(w: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # scalar per-entry importance; returns real tensor same shape as w
        if torch.is_complex(w):
            # s = |Re(w)*Re(g) + Im(w)*Im(g)|
            return (w.real * g.real + w.imag * g.imag).abs()
        return (w * g).abs()

    # symbolic layers
    for layer in model.symbolic_layers:
        w = layer.weights
        m = layer.mask
        g = layer.weights.grad
        if g is None:
            continue

        active = (m > 0.5)
        n_active += int(active.sum().item())
        if active.any():
            s = _score_tensor(w, g)
            # keep only active entries
            scores.append(s[active].reshape(-1))
            # store mapping from active entries -> flat indices
            active_flat_idx = active.view(-1).nonzero(as_tuple=False).reshape(-1)
            refs.append(("sym", layer, active_flat_idx))

    # assembly
    wA = model.assembly_layer.weights
    mA = model.assembly_layer.mask
    gA = model.assembly_layer.weights.grad
    if gA is not None:
        activeA = (mA > 0.5)
        n_active += int(activeA.sum().item())
        if activeA.any():
            sA = _score_tensor(wA, gA)
            scores.append(sA[activeA].reshape(-1))
            activeA_flat_idx = activeA.view(-1).nonzero(as_tuple=False).reshape(-1)
            refs.append(("asm", model.assembly_layer, activeA_flat_idx))

    if n_active <= int(min_edges_total):
        model.zero_grad(set_to_none=True)
        return 0

    all_scores = torch.cat(scores, dim=0)
    n = int(all_scores.numel())
    k_target = int(np.floor(frac * n))
    k_cap = n - int(min_edges_total)
    k = max(0, min(k_target, k_cap))
    if k <= 0:
        model.zero_grad(set_to_none=True)
        return 0

    # pick k smallest scores
    # (largest=False returns k smallest)
    kth = torch.topk(all_scores, k, largest=False).indices

    # map global indices back to (module, flat_idx)
    # build prefix offsets
    sizes = [0]
    for s in scores:
        sizes.append(sizes[-1] + int(s.numel()))

    pruned = 0
    for t in kth.tolist():
        # locate which ref bucket
        bi = 0
        while not (sizes[bi] <= t < sizes[bi + 1]):
            bi += 1
        local = t - sizes[bi]

        kind, obj, active_flat_idx = refs[bi]
        flat_idx = active_flat_idx[local].view(1)

        if kind == "sym":
            _apply_prune_mask_entries_(obj.mask.data, obj.weights.data, flat_idx)
        else:
            _apply_prune_mask_entries_(obj.mask.data, obj.weights.data, flat_idx)

        pruned += 1

    # cleanup numerics
    model.sanitize_weights(clamp_value=float(getattr(cfg, "clamp_limit", 1e15)))

    model.zero_grad(set_to_none=True)
    return pruned

@torch.no_grad()
def ablation_prune_on_batch(
    *,
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn,
    cfg,
    frac: float,
    device: torch.device | str,
    op_params=None,
    pred_abs_max: float | None = None,
    min_edges_total: int = 0,
    use_baseline_valid_mask: bool = True,
    print_stats: bool = False,
) -> int:
    """
    True ablation pruning:
    for each active edge, set it to zero, recompute batch loss, measure Δ = L_i - L0.
    Prune frac of edges with smallest Δ (least harmful or helpful).

    Returns number of edges pruned (mask entries zeroed).
    """
    device = torch.device(device)
    frac = float(frac)
    if frac <= 0.0:
        return 0

    model.eval()

    X = X.to(device)
    y = y.to(device)

    if pred_abs_max is None:
        pred_abs_max = float(getattr(cfg, "pred_abs_max", 1e20))

    # ---- baseline ----
    model.clear_unary_input_cache_()
    pred0 = model(X, op_params=op_params)
    valid0 = finite_mask_pred_and_target(pred0, y, pred_abs_max=float(pred_abs_max))
    n_valid0 = int(valid0.sum().item())
    if n_valid0 == 0:
        return 0

    L0 = loss_fn(pred0.real[valid0], y[valid0]).detach()

    # collect all active entries as (kind, module, flat_idx)
    entries: list[tuple[str, object, int]] = []

    for layer in model.symbolic_layers:
        active = (layer.mask > 0.5).view(-1)
        idxs = active.nonzero(as_tuple=False).view(-1).tolist()
        for fi in idxs:
            entries.append(("sym", layer, int(fi)))

    activeA = (model.assembly_layer.mask > 0.5).view(-1)
    idxsA = activeA.nonzero(as_tuple=False).view(-1).tolist()
    for fi in idxsA:
        entries.append(("asm", model.assembly_layer, int(fi)))

    n_active = len(entries)
    if n_active <= int(min_edges_total):
        return 0

    k_target = int(np.floor(frac * n_active))
    k_cap = n_active - int(min_edges_total)
    k = max(0, min(k_target, k_cap))
    if k <= 0:
        return 0
    if print_stats:
        print(f"[ABL_STATS] active_before={n_active} will_prune_k={k} keep_after={n_active - k}")

    def _get_views(obj):
        return obj.mask.view(-1), obj.weights.view(-1)

    deltas = torch.empty(n_active, device=device, dtype=L0.dtype)

    for t, (_kind, obj, flat_idx) in enumerate(entries):
        m, w = _get_views(obj)

        w_old = w[flat_idx].clone()
        m_old = m[flat_idx].clone()

        # ablate: set weight to zero (mask unchanged)
        w[flat_idx] = torch.complex(w_old.real.new_zeros(()), w_old.real.new_zeros(()))

        model.clear_unary_input_cache_()
        pred1 = model(X, op_params=op_params)

        if use_baseline_valid_mask:
            valid = valid0
            bad = ~torch.isfinite(pred1.real[valid])
            if bad.any():
                delta = torch.tensor(float("inf"), device=device, dtype=L0.dtype)
            else:
                L1 = loss_fn(pred1.real[valid], y[valid]).detach()
                delta = L1 - L0
        else:
            valid1 = finite_mask_pred_and_target(pred1, y, pred_abs_max=float(pred_abs_max))
            if int(valid1.sum().item()) == 0:
                delta = torch.tensor(float("inf"), device=device, dtype=L0.dtype)
            else:
                L1 = loss_fn(pred1.real[valid1], y[valid1]).detach()
                delta = L1 - L0

        deltas[t] = delta

        # restore
        w[flat_idx] = w_old
        m[flat_idx] = m_old

    # prune k smallest deltas
    idx_prune = torch.topk(deltas, k, largest=False).indices.tolist()

    pruned_deltas = deltas[idx_prune]  # shape (k,)
    delta_threshold = pruned_deltas.max().item()  # cutoff used
    delta_min = pruned_deltas.min().item()
    delta_mean = pruned_deltas.mean().item()

    if print_stats:
        print(
            f"[ABL_STATS] L0={float(L0.item()):.6e} "
            f"k={k}/{n_active} frac={k/n_active:.3f} "
            f"delta_thr={delta_threshold:.6e} "
            f"delta_min={delta_min:.6e} delta_mean={delta_mean:.6e}"
        )

    pruned = 0
    for t in idx_prune:
        _, obj, flat_idx = entries[t]
        m, w = _get_views(obj)
        if m[flat_idx] > 0.5:
            m[flat_idx] = 0.0
            w[flat_idx] = torch.complex(w[flat_idx].real.new_zeros(()), w[flat_idx].real.new_zeros(()))
            pruned += 1

    model.sanitize_weights(clamp_value=float(getattr(cfg, "clamp_limit", 1e15)))

    if print_stats:
        active_after = 0
        for layer in model.symbolic_layers:
            active_after += int((layer.mask > 0.5).sum().item())
        active_after += int((model.assembly_layer.mask > 0.5).sum().item())
        print(f"[ABL_STATS] pruned={pruned} active_after={active_after}")

    return pruned


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
    l1_enabled: bool,
    l1_use_real_only: bool,
    l1_coeff: float,
    l1_eps: float,
    imag_weights_penalty_enabled: bool,
    imag_weights_penalty_coeff: float,
    prune_now: bool,
    prune_threshold: float,
    normalize_divisions: bool = False,
    normalize_divisions_eps: float = 1e-12,
    clamp_pred: bool = True,
    clamp_limit: float = 1e15,
    imag_shrink_enabled: bool = False,
    imag_shrink_coeff: float = 1.0,
    theta_penalty_enabled: bool,
    theta_penalty_coeff: float,
    theta_penalty_eps: float = 1e-12,
):
    model.train()
    device = torch.device(device)

    op_params_epoch = op_params_override if (op_params_override is not None) else build_op_params(epoch, cfg)

    total_data_loss = 0.0
    total_reg_loss = 0.0
    total_real_reg_loss = 0.0
    total_imag_w_reg_loss = 0.0
    total_theta_reg_loss = 0.0
    total_loss = 0.0

    # NEW: counters for valid fraction
    num_points_total = 0
    num_points_valid = 0

    for _, (X, y) in enumerate(dataloader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        model.clear_unary_input_cache_()
        pred = model(X, op_params=op_params_epoch)

        pred_abs_max = cfg.pred_abs_max
        valid = finite_mask_pred_and_target(pred, y, pred_abs_max=pred_abs_max)

        # NEW: count attempted points (same shape as valid)
        num_points_total += int(valid.numel())

        n_valid = int(valid.sum().item())
        num_points_valid += n_valid

        if n_valid == 0:
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

        if theta_penalty_enabled and theta_penalty_coeff > 0.0:
            theta_reg = float(theta_penalty_coeff) * nonneg_real_input_penalty_fast(
                model,
                eps=0.0,          # or cfg.nonneg_real_eps if you add it
                squared=True,     # or cfg.nonneg_real_squared
            )
        else:
            theta_reg = pred.real.new_tensor(0.0)

        reg_loss = real_reg + imag_w_reg + theta_reg
        loss = data_loss + reg_loss

        loss.backward()
        model.sanitize_gradients(max_grad=1e8)
        optimizer.step()

        bs = n_valid
        total_data_loss += data_loss.item() * bs
        total_reg_loss += reg_loss.item() * bs
        total_real_reg_loss += real_reg.item() * bs
        total_imag_w_reg_loss += imag_w_reg.item() * bs
        total_theta_reg_loss += theta_reg.item() * bs
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

    denom = max(1, num_points_valid)  # keep your loss-averaging behavior

    valid_frac = (num_points_valid / max(1, num_points_total))

    return (
        total_loss / denom,
        total_data_loss / denom,
        total_reg_loss / denom,
        total_real_reg_loss / denom,
        total_imag_w_reg_loss / denom,
        total_theta_reg_loss / denom,
        op_params_epoch,
        valid_frac,
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
    on_print: Optional[Callable[[int, torch.nn.Module], None]] = None,
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
            )
        if isinstance(old_opt, torch.optim.AdamW):
            return torch.optim.AdamW(
                new_model.parameters(),
                lr=old_opt.param_groups[0]["lr"],
            )
        if isinstance(old_opt, torch.optim.RMSprop):
            return torch.optim.RMSprop(
                new_model.parameters(),
                lr=old_opt.param_groups[0]["lr"],
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

    def _maybe_print(tag: str, avg_total: float, avg_data: float, avg_sparse: float, avg_imag_w: float, avg_theta: float, valid_frac: float):
        if (global_epoch + 1) % cfg.print_every == 0 or global_epoch == 0:
            lr = opt.param_groups[0]["lr"]
            active = model.count_active_edges()
            print(
                f"[{tag} | Epoch {global_epoch+1}] "
                f"lr={lr:.2e}, total={avg_total:.4e}, data={avg_data:.4e}, "
                f"sparsity_reg={avg_sparse:.4e}, imag_w={avg_imag_w:.4e}, "
                f"theta={avg_theta:.4e}, "
                f"valid={valid_frac:.3f}, "
                f"active_edges={active}"
            )

            if on_print is not None:
                on_print(global_epoch + 1, model)

    thr_min = float(getattr(cfg, "pruning_threshold_min", 0.0))
    thr_max = float(getattr(cfg, "pruning_threshold_max", float("inf")))
    min_edges_layer = int(getattr(cfg, "pruning_min_edges_per_layer", 0))

    def _get_one_batch():
        xb, yb = next(iter(dataloader))
        return xb.to(device), yb.to(device)
    
    
    def _do_prune_and_rebuild(tag: str, prune_fraction: float):
        nonlocal model, opt, sch

        # --- NEW: ablation pruning on one batch ---
        if bool(getattr(cfg, "ablation_prune_enabled", False)) and prune_fraction > 0.0:
            xb, yb = _get_one_batch()
            before_edges = model.count_active_edges()
            pruned_abl = ablation_prune_on_batch(
                model=model,
                X=xb,
                y=yb,
                loss_fn=loss_fn,
                cfg=cfg,
                frac=float(getattr(cfg, "ablation_prune_fraction_phase2", prune_fraction)),
                device=device,
                op_params=None,
                pred_abs_max=float(getattr(cfg, "pred_abs_max", 1e20)),
                min_edges_total=int(getattr(cfg, "ablation_min_edges_total", 0)),
                use_baseline_valid_mask=True,
                print_stats=True,
            )
            if pruned_abl > 0:
                after_edges = model.count_active_edges()
                print(f"[ABL_PRUNE | {tag}] pruned={pruned_abl} active {before_edges}->{after_edges}")

        # keep your existing pipeline:
        dropped_div = model.drop_div_ops_with_pruned_denominator_()
        if dropped_div > 0:
            print(f"[DROP_DIV | {tag}] pruned_downstream_edges={dropped_div}")

        dropped_log = model.drop_log_ops_with_pruned_input_()
        if dropped_log > 0:
            print(f"[DROP_LOG | {tag}] pruned_downstream_edges={dropped_log}")

        cleaned = model.cascade_cleanup_disconnected_()
        if cleaned > 0:
            print(f"[CLEAN | {tag}] cleaned_disconnected={cleaned}")

        before_rebuild_total = model.count_active_edges()
        model = model.rebuild_from_pruned().to(device)
        opt = _rebuild_optimizer_like(opt, model)
        sch = _rebuild_scheduler_like(sch, opt, cfg)
        after_total = model.count_active_edges()
        if after_total != before_rebuild_total:
            print(f"[REBUILD | {tag}] active {before_rebuild_total}->{after_total}")

    # -------------------------
    # Phase 1
    # -------------------------
    phase1_epochs = int(getattr(cfg, "phase1_epochs", 0))
    for _ in range(phase1_epochs):
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, avg_theta, _, valid_frac = train_one_epoch(
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
            imag_shrink_enabled=cfg.phase1_imag_shrink_enabled,
            imag_shrink_coeff=cfg.phase1_imag_shrink_coeff,
            theta_penalty_enabled=True,
            theta_penalty_coeff=float(getattr(cfg, "theta_coeff_phase1", 0.0)),
            theta_penalty_eps=float(getattr(cfg, "theta_eps", 1e-12)),
        )
        _maybe_print("PHASE1", avg_total, avg_data, avg_sparse, avg_imag_w, avg_theta, valid_frac)
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

        dropped_div = model.drop_div_ops_with_pruned_denominator_()
        dropped_log = model.drop_log_ops_with_pruned_input_()
        cleaned = model.cascade_cleanup_disconnected_()

        if dropped_div > 0:
            print(f"[DROP_DIV | {tag}] pruned_downstream_edges={dropped_div}")
        if dropped_log > 0:
            print(f"[DROP_LOG | {tag}] pruned_downstream_edges={dropped_log}")
        if cleaned > 0:
            print(f"[CLEAN | {tag}] cleaned_disconnected={cleaned}")

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
    normalize_divs = bool(getattr(cfg, "normalize_divisions_phase2", True))

    for e in range(phase2_epochs):
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, avg_theta, _, valid_frac = train_one_epoch(
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
            imag_shrink_enabled=cfg.phase2_imag_shrink_enabled,
            imag_shrink_coeff=cfg.phase2_imag_shrink_coeff,
            theta_penalty_enabled=True,
            theta_penalty_coeff=float(getattr(cfg, "theta_coeff_phase2", 0.0)),
            theta_penalty_eps=float(getattr(cfg, "theta_eps", 1e-12)),
        )
        _maybe_print("PHASE2", avg_total, avg_data, avg_sparse, avg_imag_w, avg_theta, valid_frac)
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
    normalize_divs = bool(getattr(cfg, "normalize_divisions_phase3", True))
    if bool(getattr(cfg, "phase3_force_real", False)):
        # normalize first so projection doesn't freeze a bad scaling
        model.normalize_all_divisions_(eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)))
        model.force_real_()
        model.freeze_imag_()

    for _ in range(phase3_epochs):
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, avg_theta, _, valid_frac = train_one_epoch(
            epoch=global_epoch,
            dataloader=dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=opt,
            device=device,
            cfg=cfg,
            op_params_override=None,
            l1_enabled=bool(getattr(cfg, "phase3_l1_enabled", False)),
            l1_use_real_only=cfg.l1_on_real_only,
            l1_coeff=float(getattr(cfg, "l1_reg_coeff_phase3", 0.0)),
            l1_eps=float(getattr(cfg, "l1_eps", 1e-12)),
            imag_weights_penalty_enabled=True,
            imag_weights_penalty_coeff=float(getattr(cfg, "imag_w_coeff_phase3", 0.0)),
            prune_now=False,
            prune_threshold=0.0,
            normalize_divisions=normalize_divs,
            normalize_divisions_eps=cfg.normalize_divisions_eps,
            clamp_pred=getattr(cfg, "clamp_pred", True),
            clamp_limit=getattr(cfg, "clamp_limit", 1e15),
            imag_shrink_enabled=cfg.phase3_imag_shrink_enabled,
            imag_shrink_coeff=cfg.phase3_imag_shrink_coeff,
            theta_penalty_enabled=True,
            theta_penalty_coeff=float(getattr(cfg, "theta_coeff_phase3", 0.0)),
            theta_penalty_eps=float(getattr(cfg, "theta_eps", 1e-12)),
        )

        if sch is not None:
            try:
                sch.step(avg_data)
            except TypeError:
                sch.step()

        _maybe_print("PHASE3", avg_total, avg_data, avg_sparse, avg_imag_w, avg_theta, valid_frac)
        _record(avg_data, avg_imag_w)
        global_epoch += 1

    if bool(getattr(cfg, "phase3_force_real", False)):
        # normalize first so projection doesn't freeze a bad scaling
        model.normalize_all_divisions_(eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)))
        model.force_real_()
        model.freeze_imag_()
        
    return model, (imag_w_losses, data_losses)