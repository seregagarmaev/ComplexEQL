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
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def timeit(func: Callable) -> Callable:
    """
    Simple timing decorator.
    """
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
    """
    Modified L1 penalty with near-zero smoothing and L0-like saturation.
    """

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
    """
    L1L0-like penalty:
      - smooth around 0 to keep gradients well-behaved
      - saturates for large |x| like an L0 count
    """
    if isinstance(input_tensor, (list, tuple)):
        return sum(l1l0_smooth(t, alpha=alpha, s=s, eps=eps) for t in input_tensor)

    x = input_tensor
    absx = torch.abs(x)

    # Smooth |x| near 0 with a quartic polynomial (C1-continuous)
    inner = absx < s
    poly = (x**4) / (-8.0 * s**3) + (x**2) * (3.0 / (4.0 * s)) + 3.0 * s / 8.0
    smooth_abs = torch.where(inner, poly, absx)

    # L1L0-style penalty
    penalty = ((alpha + 1.0) * smooth_abs) / (alpha + smooth_abs + eps)

    return torch.sum(penalty)


# -------------------------
# Alpha scheduling (exponential)
# -------------------------
def compute_alpha(
    epoch: int,
    l1l0_start_epoch: int,
    l1l0_end_epoch: int,
    alpha_start: float,
    alpha_end: float,
) -> tuple[float, bool]:
    """
    Compute alpha_t for L1L0 and whether to use regularization
    in the given epoch.

    Exponential interpolation between alpha_start and alpha_end
    on [l1l0_start_epoch, l1l0_end_epoch].
    """
    if epoch < l1l0_start_epoch:
        # no sparsity yet
        return alpha_start, False

    if epoch <= l1l0_end_epoch:
        t_rel = epoch - l1l0_start_epoch
        T_rel = max(1, l1l0_end_epoch - l1l0_start_epoch)
        decay_ratio = t_rel / T_rel
        alpha_t = alpha_start * ((alpha_end / alpha_start) ** decay_ratio)
        return alpha_t, True

    # after end epoch, keep alpha at final value
    return alpha_end, True


def _anneal_coeff(
    epoch_in_cycle: int,
    anneal_epochs: int,
    start: float,
    end: float,
    mode: str,
) -> float:
    if anneal_epochs <= 0:
        return float(end)

    t = min(max(epoch_in_cycle, 0), anneal_epochs)
    r = t / max(1, anneal_epochs)

    if mode == "linear":
        return float(start + (end - start) * r)

    if mode == "exp":
        if start <= 0 or end <= 0:
            return float(start + (end - start) * r)
        return float(start * ((end / start) ** r))

    raise ValueError(f"Unknown anneal mode: {mode!r}")


@torch.no_grad()
def reinit_imag_weights_(model: torch.nn.Module, scale: float) -> None:
    """Reinitialize Im(weights) ~ U(-scale/2, scale/2) for ACTIVE (mask==1) entries only."""
    for layer in model.symbolic_layers:
        w = layer.weights.data
        m = layer.mask.data.bool()
        new_imag = (torch.rand_like(w.imag) - 0.5) * scale
        imag = torch.where(m, new_imag, torch.zeros_like(new_imag))
        layer.weights.data = torch.complex(w.real, imag)

    w = model.assembly_layer.weights.data
    m = model.assembly_layer.mask.data.bool()
    new_imag = (torch.rand_like(w.imag) - 0.5) * scale
    imag = torch.where(m, new_imag, torch.zeros_like(new_imag))
    model.assembly_layer.weights.data = torch.complex(w.real, imag)

    

def train_one_epoch(
    epoch: int,
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    alpha_t: float,
    sparsity_enabled: bool,
    l1l0_use_real_only: bool,
    imag_weights_penalty_enabled: bool,
    l1l0_coeff: float,
    imag_weights_penalty_coeff: float,
    l1l0_s: float,
    l1l0_eps: float,
    prune_now: bool,
    prune_threshold: float,
    normalize_divisions: bool = False,
    normalize_divisions_eps: float = 1e-12,
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

        data_loss = loss_fn(pred.real, y)

        # ----------------- L1L0 sparsity -----------------
        if sparsity_enabled and l1l0_coeff > 0.0:
            if l1l0_use_real_only:
                reg_target = model.get_real_weights_list()
            else:
                real_list = model.get_real_weights_list()
                imag_list = model.get_imag_weights_list()
                reg_target = [torch.sqrt(r**2 + i**2) for r, i in zip(real_list, imag_list)]

            real_reg = l1l0_coeff * l1l0_penalty(reg_target, alpha=alpha_t, s=l1l0_s, eps=l1l0_eps)
        else:
            real_reg = torch.tensor(0.0, device=device)

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
        model.sanitize_gradients(max_grad=1e3)
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
        pruned = model.prune_by_threshold(prune_threshold)
        if pruned > 0:
            print(f"[prune] Epoch {epoch}: pruned {pruned} weights (|w| < {prune_threshold:g})")

    # normalize division mixing once per epoch
    if normalize_divisions:
        n_norm = model.normalize_all_divisions_(eps=normalize_divisions_eps)
    
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
    optimizer: torch.optim.Optimizer,  # ignored
    loss_fn: Callable,
    cfg,
    device: torch.device | str,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,  # ignored
):
    device = torch.device(device)
    global_epoch = 0

    sl0_weights, sl1_weights, al_weights = [], [], []
    data_losses, imag_w_losses = [], []

    def _make_scheduler(opt):
        if cfg.scheduler == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **cfg.schedulerparams)
        return None

    def _plateau_update(
        best: float, no_improve: int, current: float, rel_tol: float
    ) -> tuple[float, int]:
        # improvement if current <= best * (1 - rel_tol)
        if current <= best * (1.0 - rel_tol):
            return current, 0
        return best, no_improve + 1

    # ----------------------------------------------------------
    # Single optimizer + scheduler for all phases.
    # Scheduler is reset ONLY when we reinit imaginary weights.
    # ----------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sch = None # _make_scheduler(opt)

    def _run_phase(
        phase_name: str,
        num_epochs: int,
        *,
        l1l0_enabled: bool,
        l1l0_coeff: float,
        l1l0_use_real_only: bool,
        pruning_enabled: bool,
        imag_enabled: bool,
        do_resets: bool,
        imag_coeff_fixed: float | None,
        phase2_scheduler_warmup: int = 0,
    ):
        nonlocal global_epoch, sl0_weights, al_weights, data_losses, imag_w_losses
        nonlocal opt, sch

        # cycle state for imag anneal/reset
        cycle_epoch = 0
        best_data = float("inf")
        no_improve = 0

        for e in range(num_epochs):
            # imag coefficient: fixed (phase 3) or annealed per cycle (phases 1/2)
            if imag_coeff_fixed is not None:
                imag_coeff = float(imag_coeff_fixed)
            else:
                imag_coeff = _anneal_coeff(
                    epoch_in_cycle=cycle_epoch,
                    anneal_epochs=cfg.imag_anneal_epochs,
                    start=cfg.imag_w_coeff_start,
                    end=cfg.imag_w_coeff_end,
                    mode=cfg.imag_anneal_mode,
                )

            # sparsity schedule
            if l1l0_enabled and l1l0_coeff > 0.0:
                alpha_t, sparsity_enabled = compute_alpha(
                    e,
                    l1l0_start_epoch=cfg.l1l0_start_epoch,
                    l1l0_end_epoch=cfg.l1l0_end_epoch,
                    alpha_start=cfg.alpha_start,
                    alpha_end=cfg.alpha_end,
                )
            else:
                alpha_t, sparsity_enabled = 0.0, False

            # pruning schedule
            prune_now = False
            if pruning_enabled and cfg.pruning_threshold > 0.0:
                if e >= cfg.pruning_start_epoch and (e - cfg.pruning_start_epoch) % cfg.pruning_period == 0:
                    prune_now = True

            avg_total, avg_data, _avg_reg, avg_real_reg, avg_imag_w_reg = train_one_epoch(
                epoch=global_epoch,
                dataloader=dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=opt,  # persistent optimizer
                device=device,
                alpha_t=alpha_t,
                sparsity_enabled=sparsity_enabled,
                l1l0_use_real_only=l1l0_use_real_only,
                imag_weights_penalty_enabled=imag_enabled,
                l1l0_coeff=l1l0_coeff,
                imag_weights_penalty_coeff=imag_coeff if imag_enabled else 0.0,
                l1l0_s=cfg.l1l0_s,
                l1l0_eps=cfg.l1l0_eps,
                prune_now=prune_now,
                prune_threshold=cfg.pruning_threshold if prune_now else 0.0,
                normalize_divisions=cfg.normalize_divisions,
                normalize_divisions_eps=cfg.normalize_divisions_eps,
            )

            # scheduler (persistent; reset only on imag reinit)
            # if sch is not None:
            #     if phase_name == "Phase 2" and e < phase2_scheduler_warmup:
            #         pass
            #     else:
            #         sch.step(avg_total)
            if (phase_name == "Phase 3") and (sch is not None):
                sch.step(avg_data)

            if (global_epoch + 1) % cfg.print_every == 0 or global_epoch == 0:
                lr = opt.param_groups[0]["lr"]
                print(
                    f"[{phase_name} | Epoch {global_epoch+1}] "
                    f"total={avg_total:.4e}, data={avg_data:.4e}, "
                    f"sparsity_reg={avg_real_reg:.4e}, imag_w={avg_imag_w_reg:.4e}, "
                    f"alpha={alpha_t:.3e}, imag_coeff={imag_coeff:.3e}, lr={lr:.2e}"
                )

            # plateau -> reset imag weights & restart anneal cycle (phases 1/2)
            if do_resets and imag_enabled:
                if e >= cfg.imag_plateau_min_epoch_in_phase:
                    can_check = True
                    if cfg.imag_plateau_check_after_anneal:
                        can_check = cycle_epoch >= cfg.imag_anneal_epochs

                    if can_check:
                        best_data, no_improve = _plateau_update(
                            best_data, no_improve, avg_data, cfg.imag_plateau_rel_tol
                        )
                        if no_improve >= cfg.imag_plateau_patience:
                            # 1) reinit imaginary weights
                            reinit_imag_weights_(model, cfg.imag_reinit_scale)

                            # 2) reset optimizer LR to initial value
                            for g in opt.param_groups:
                                g["lr"] = cfg.lr

                            # 3) re-create scheduler (fresh state)
                            if sch is not None:
                                sch = _make_scheduler(opt)

                            # 4) restart cycle
                            cycle_epoch = 0
                            best_data = float("inf")
                            no_improve = 0
                        else:
                            cycle_epoch += 1
                    else:
                        cycle_epoch += 1
                else:
                    cycle_epoch += 1
            else:
                cycle_epoch += 1

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
        l1l0_coeff=0.0,
        l1l0_use_real_only=cfg.l1l0_on_real_only,
        pruning_enabled=cfg.pruning_enabled_phase1,
        imag_enabled=cfg.imag_weights_penalty_enabled_phase1,
        do_resets=True,
        imag_coeff_fixed=None,
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
        pruning_enabled=cfg.pruning_enabled_phase2,
        imag_enabled=cfg.imag_weights_penalty_enabled_phase2,
        do_resets=True,
        imag_coeff_fixed=None,
        phase2_scheduler_warmup=cfg.scheduler_warmup_phase2,
    )

    # --------------------------
    # PHASE 3
    # --------------------------
    sch = _make_scheduler(opt)
    _run_phase(
        "Phase 3",
        cfg.phase3_epochs,
        l1l0_enabled=cfg.l1l0_enabled_phase3,
        l1l0_coeff=0.0,
        l1l0_use_real_only=cfg.l1l0_on_real_only,
        pruning_enabled=cfg.pruning_enabled_phase3,
        imag_enabled=cfg.imag_weights_penalty_enabled_phase3,
        do_resets=False,
        imag_coeff_fixed=cfg.imag_w_coeff_phase3,
    )

    return model, (sl0_weights, sl1_weights, al_weights, imag_w_losses, data_losses)

