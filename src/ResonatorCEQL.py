from __future__ import annotations

from copy import copy
from typing import Any, Callable, Dict, Iterable, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sympy as sp

from src.operations import OpParams
from src.ComplexEQL import ComplexEQL, _sanitize_symbolic
from src.sympy_utils import prune_small_coeff_terms
from src.utils import build_op_params


class ResonatorBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        eql_cfg = copy(cfg)
        eql_cfg.n_input_fields = int(cfg.n_input_fields) - 1

        self.eql = ComplexEQL(eql_cfg)

        self.A = nn.Parameter(torch.complex(torch.rand(1), torch.rand(1)))
        self.gamma = nn.Parameter(torch.complex(torch.rand(1), torch.rand(1)))
    
    def forward(self, x: torch.Tensor, *, op_params: Optional[OpParams] = None,) -> torch.Tensor:
        omega = x[:, 0:1]

        x_aux = x[:, 1:]
        h = self.eql(x_aux, op_params=op_params)
        
        num = self.A
        den = (omega - h) ** 2 + self.gamma
        return num / den
    

class LinearResonatorWrapper(nn.Module):
    def __init__(self, cfg, n_resonators: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.n_resonators = int(cfg.n_resonators) if n_resonators is None else int(n_resonators)

        self.a = nn.Parameter(torch.complex(torch.rand(1), torch.rand(1)))
        self.b = nn.Parameter(torch.complex(torch.rand(1), torch.rand(1)))

        self.resonators = nn.ModuleList(
            [ResonatorBlock(cfg) for _ in range(self.n_resonators)]
        )
    
    def _is_dead_resonator(self, r: ResonatorBlock) -> bool:
        return r.eql.count_active_edges() == 0

    def forward(self, x: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
        omega = x[:, 0:1].to(self.a.dtype)
        y = self.a * omega + self.b

        for r in self.resonators:
            y = y + r(x, op_params=op_params)

        return y

    def clear_unary_input_cache_(self) -> None:
        for r in self.resonators:
            r.eql.clear_unary_input_cache_()

    def get_unary_penalty_inputs(self, *, detach: bool = False):
        outs = []
        for r in self.resonators:
            outs.extend(r.eql.get_unary_penalty_inputs(detach=detach))
        return outs

    @torch.no_grad()
    def count_active_edges(self) -> int:
        total = 0
        for r in self.resonators:
            total += r.eql.count_active_edges()
        return total

    @torch.no_grad()
    def count_active_edges_per_resonator(self):
        return [r.eql.count_active_edges() for r in self.resonators]

    @torch.no_grad()
    def normalize_all_divisions_(self, eps: float = 1e-12) -> int:
        total = 0
        for r in self.resonators:
            total += r.eql.normalize_all_divisions_(eps=eps)
        return total

    @torch.no_grad()
    def drop_div_ops_with_pruned_denominator_(self) -> int:
        total = 0
        for r in self.resonators:
            total += r.eql.drop_div_ops_with_pruned_denominator_()
        return total

    @torch.no_grad()
    def drop_log_ops_with_pruned_input_(self) -> int:
        total = 0
        for r in self.resonators:
            total += r.eql.drop_log_ops_with_pruned_input_()
        return total

    @torch.no_grad()
    def cascade_cleanup_disconnected_(self) -> int:
        total = 0
        for r in self.resonators:
            total += r.eql.cascade_cleanup_disconnected_()
        return total

    def sanitize_gradients(self, max_grad: float = 1e3) -> None:
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is None:
                    continue
                g = p.grad.data
                if torch.is_complex(g):
                    g = torch.complex(
                        torch.nan_to_num(g.real, nan=0.0, posinf=0.0, neginf=0.0),
                        torch.nan_to_num(g.imag, nan=0.0, posinf=0.0, neginf=0.0),
                    )
                    mag = torch.abs(g)
                    mask = mag > max_grad
                    if mask.any():
                        g = g.clone()
                        g[mask] = g[mask] * (max_grad / mag[mask])
                else:
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                    g.clamp_(-max_grad, max_grad)
                p.grad.data.copy_(g)

    def sanitize_weights(self, clamp_value: float = 1e6) -> None:
        with torch.no_grad():
            for p in self.parameters():
                w = p.data
                if torch.is_complex(w):
                    w = torch.complex(
                        torch.nan_to_num(w.real, nan=0.0, posinf=0.0, neginf=0.0),
                        torch.nan_to_num(w.imag, nan=0.0, posinf=0.0, neginf=0.0),
                    )
                    mag = torch.abs(w)
                    mask = mag > clamp_value
                    if mask.any():
                        w = w.clone()
                        w[mask] = w[mask] * (clamp_value / mag[mask])
                else:
                    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                    w.clamp_(-clamp_value, clamp_value)
                p.data.copy_(w)

    @torch.no_grad()
    def shrink_imag_weights_(self, coeff: float) -> None:
        coeff = float(coeff)
        if coeff >= 1.0:
            return
        if coeff < 0.0:
            raise ValueError("shrink coeff must be in [0,1].")

        for p in self.parameters():
            if torch.is_complex(p.data):
                p.data = torch.complex(p.data.real, p.data.imag * coeff)

    @torch.no_grad()
    def force_real_(self) -> None:
        for p in self.parameters():
            if torch.is_complex(p.data):
                p.data = torch.complex(p.data.real, torch.zeros_like(p.data.imag))

    def freeze_imag_(self) -> None:
        def _hook_zero_imag(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            if torch.is_complex(grad):
                return torch.complex(grad.real, torch.zeros_like(grad.imag))
            return grad

        for p in self.parameters():
            if torch.is_complex(p):
                p.register_hook(_hook_zero_imag)

    @torch.no_grad()
    def rebuild_from_pruned(self) -> "LinearResonatorWrapper":
        alive_old_resonators = [r for r in self.resonators if r.eql.count_active_edges() > 0]

        new_model = LinearResonatorWrapper(
            self.cfg,
            n_resonators=len(alive_old_resonators),
        ).to(self.a.device)

        new_model.a.data.copy_(self.a.data)
        new_model.b.data.copy_(self.b.data)

        for new_r, old_r in zip(new_model.resonators, alive_old_resonators):
            new_r.A.data.copy_(old_r.A.data)
            new_r.gamma.data.copy_(old_r.gamma.data)
            new_r.eql = old_r.eql.rebuild_from_pruned().to(self.a.device)

        return new_model
    
    def get_symbolic_expression(
        self,
        symbolic_inputs: List[sp.Expr],
        rounding_decimals: int = 2,
        *,
        use_imag: bool = False,
    ) -> sp.Expr:
        omega = symbolic_inputs[0]
        aux_inputs = symbolic_inputs[1:]

        def _coef(w: torch.Tensor) -> sp.Expr:
            wr = sp.Float(float(w.real.detach().cpu().item()))
            if not use_imag:
                return wr
            wi = sp.Float(float(w.imag.detach().cpu().item()))
            return wr + sp.I * wi

        expr: sp.Expr = _coef(self.a) * omega + _coef(self.b)

        for block in self.resonators:
            h = block.eql.get_symbolic_expression(
                aux_inputs,
                rounding_decimals=rounding_decimals,
                use_imag=use_imag,
            )

            A = _coef(block.A)
            gamma = _coef(block.gamma)

            expr = expr + A / ((omega - h) ** 2 + gamma)

        expr = _sanitize_symbolic(expr)
        return prune_small_coeff_terms(expr, rounding_decimals)
    

# ============================================================
# Wrapper-aware helpers
# ============================================================

def get_all_symbolic_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    layers = []
    for r in model.resonators:
        layers.extend(r.eql.symbolic_layers)
    return layers


def l1_reg_fast_wrapper(
    model: torch.nn.Module,
    *,
    use_real_only: bool,
    eps: float,
) -> torch.Tensor:
    acc: torch.Tensor | None = None

    for layer in get_all_symbolic_layers(model):
        w = layer.weights * layer.mask.to(layer.weights.dtype)
        term = (
            w.real.abs().sum()
            if use_real_only
            else torch.sqrt(w.real * w.real + w.imag * w.imag + eps).sum()
        )
        acc = term if acc is None else (acc + term)

    if acc is None:
        p = next(model.parameters())
        base = p.real if torch.is_complex(p) else p
        return base.new_tensor(0.0)

    return acc


def imag_w_l2_reg_fast_wrapper(model: torch.nn.Module) -> torch.Tensor:
    acc: torch.Tensor | None = None

    for layer in get_all_symbolic_layers(model):
        w = layer.weights * layer.mask.to(layer.weights.dtype)
        term = (w.imag * w.imag).sum()
        acc = term if acc is None else (acc + term)

    if acc is None:
        p = next(model.parameters())
        base = p.real if torch.is_complex(p) else p
        return base.new_tensor(0.0)

    return acc


def nonneg_real_input_penalty_fast(
    model: torch.nn.Module,
    *,
    eps: float = 0.0,
    squared: bool = True,
) -> torch.Tensor:
    infos = model.get_unary_penalty_inputs(detach=False)
    if len(infos) == 0:
        p = next(model.parameters())
        base = p.real if torch.is_complex(p) else p
        return base.new_tensor(0.0)

    acc = None
    cnt = 0

    for d in infos:
        z = d["z"]
        r = z.real
        mask = torch.isfinite(r)
        if eps > 0.0:
            mask = mask & (r < 1e30)

        if not mask.any():
            continue

        neg = torch.relu(-r[mask])
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
    if torch.is_complex(pred):
        pred_finite = torch.isfinite(pred.real) & torch.isfinite(pred.imag)
        pred_mag = pred.real.abs()
    else:
        pred_finite = torch.isfinite(pred)
        pred_mag = pred.abs()

    pred_ok = pred_finite & (pred_mag <= float(pred_abs_max))

    while pred_ok.dim() > 1:
        pred_ok = pred_ok.all(dim=-1)

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
    m = mask.view(-1)
    w = weights.view(-1)
    m[idx_flat] = 0.0

    if torch.is_complex(w):
        w[idx_flat] = torch.complex(
            w.real[idx_flat].new_zeros(idx_flat.shape),
            w.real[idx_flat].new_zeros(idx_flat.shape),
        )
    else:
        w[idx_flat] = w[idx_flat].new_zeros(idx_flat.shape)


def _iter_prunable_modules_wrapper(model: torch.nn.Module):
    for ridx, r in enumerate(model.resonators):
        eql = r.eql

        for lidx, layer in enumerate(eql.symbolic_layers):
            yield ("sym", ridx, lidx, layer)

        yield ("asm", ridx, None, eql.assembly_layer)


@torch.no_grad()
def ablation_prune_on_batch_wrapper(
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
    device = torch.device(device)
    frac = float(frac)
    if frac <= 0.0:
        return 0

    model.eval()
    X = X.to(device)
    y = y.to(device)

    if pred_abs_max is None:
        pred_abs_max = float(getattr(cfg, "pred_abs_max", 1e20))

    model.clear_unary_input_cache_()
    pred0 = model(X, op_params=op_params)
    valid0 = finite_mask_pred_and_target(pred0, y, pred_abs_max=float(pred_abs_max))
    n_valid0 = int(valid0.sum().item())
    if n_valid0 == 0:
        return 0

    L0 = loss_fn(pred0.real[valid0], y[valid0]).detach()

    entries: list[tuple[str, object, int]] = []

    for kind, _ridx, _lidx, obj in _iter_prunable_modules_wrapper(model):
        active = (obj.mask > 0.5).view(-1)
        idxs = active.nonzero(as_tuple=False).view(-1).tolist()
        for fi in idxs:
            entries.append((kind, obj, int(fi)))

    n_active = len(entries)
    if n_active <= int(min_edges_total):
        return 0

    k_target = int(np.floor(frac * n_active))
    k_cap = n_active - int(min_edges_total)
    k = max(0, min(k_target, k_cap))
    if k <= 0:
        return 0

    if print_stats:
        print(f"[ABL_STATS] active_before={n_active} will_prune_k={k} keep_after={n_active-k}")

    def _get_views(obj):
        return obj.mask.view(-1), obj.weights.view(-1)

    deltas = torch.empty(n_active, device=device, dtype=L0.dtype)

    for t, (_kind, obj, flat_idx) in enumerate(entries):
        m, w = _get_views(obj)

        w_old = w[flat_idx].clone()
        m_old = m[flat_idx].clone()

        w[flat_idx] = torch.complex(
            w_old.real.new_zeros(()),
            w_old.real.new_zeros(()),
        )

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

        w[flat_idx] = w_old
        m[flat_idx] = m_old

    idx_prune = torch.topk(deltas, k, largest=False).indices.tolist()

    if print_stats:
        pruned_deltas = deltas[idx_prune]
        print(
            f"[ABL_STATS] L0={float(L0.item()):.6e} "
            f"k={k}/{n_active} frac={k/n_active:.3f} "
            f"delta_thr={pruned_deltas.max().item():.6e} "
            f"delta_min={pruned_deltas.min().item():.6e} "
            f"delta_mean={pruned_deltas.mean().item():.6e}"
        )

    pruned = 0
    for t in idx_prune:
        _, obj, flat_idx = entries[t]
        m, w = _get_views(obj)
        if m[flat_idx] > 0.5:
            m[flat_idx] = 0.0
            w[flat_idx] = torch.complex(
                w[flat_idx].real.new_zeros(()),
                w[flat_idx].real.new_zeros(()),
            )
            pruned += 1

    model.sanitize_weights(clamp_value=float(getattr(cfg, "clamp_limit", 1e15)))

    if print_stats:
        active_after = model.count_active_edges()
        print(f"[ABL_STATS] pruned={pruned} active_after={active_after}")

    return pruned


@torch.no_grad()
def prune_by_threshold_wrapper(model: torch.nn.Module, threshold: float) -> int:
    total = 0
    for r in model.resonators:
        total += r.eql.prune_by_threshold(threshold)
    return total


# ============================================================
# One epoch
# ============================================================

def train_one_epoch_wrapper(
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
    normalize_divisions: bool = False,
    normalize_divisions_eps: float = 1e-12,
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

    num_points_total = 0
    num_points_valid = 0

    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        model.clear_unary_input_cache_()
        pred = model(X, op_params=op_params_epoch)

        valid = finite_mask_pred_and_target(
            pred,
            y,
            pred_abs_max=float(getattr(cfg, "pred_abs_max", 1e20)),
        )

        num_points_total += int(valid.numel())

        n_valid = int(valid.sum().item())
        num_points_valid += n_valid

        if n_valid == 0:
            continue

        data_loss = loss_fn(pred.real[valid], y[valid])

        if l1_enabled and l1_coeff > 0.0:
            real_reg = float(l1_coeff) * l1_reg_fast_wrapper(
                model,
                use_real_only=bool(l1_use_real_only),
                eps=float(l1_eps),
            )
        else:
            real_reg = pred.real.new_tensor(0.0)

        if imag_weights_penalty_enabled and imag_weights_penalty_coeff > 0.0:
            imag_w_reg = float(imag_weights_penalty_coeff) * imag_w_l2_reg_fast_wrapper(model)
        else:
            imag_w_reg = pred.real.new_tensor(0.0)

        if theta_penalty_enabled and theta_penalty_coeff > 0.0:
            theta_reg = float(theta_penalty_coeff) * nonneg_real_input_penalty_fast(
                model,
                eps=float(theta_penalty_eps),
                squared=True,
            )
        else:
            theta_reg = pred.real.new_tensor(0.0)

        reg_loss = real_reg + imag_w_reg + theta_reg
        loss = data_loss + reg_loss

        loss.backward()
        model.sanitize_gradients(max_grad=1e8)
        optimizer.step()
        model.sanitize_weights(clamp_value=float(getattr(cfg, "clamp_limit", 1e15)))

        bs = n_valid
        total_data_loss += data_loss.item() * bs
        total_reg_loss += reg_loss.item() * bs
        total_real_reg_loss += real_reg.item() * bs
        total_imag_w_reg_loss += imag_w_reg.item() * bs
        total_theta_reg_loss += theta_reg.item() * bs
        total_loss += loss.item() * bs

    if normalize_divisions:
        model.normalize_all_divisions_(eps=normalize_divisions_eps)

    if imag_shrink_enabled and imag_shrink_coeff < 1.0:
        model.shrink_imag_weights_(coeff=imag_shrink_coeff)

    denom = max(1, num_points_valid)
    valid_frac = num_points_valid / max(1, num_points_total)

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


# ============================================================
# Phase-based training
# ============================================================

def train_wrapper(
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
    model = model.to(device)

    global_epoch = 0
    data_losses: list[float] = []
    imag_w_losses: list[float] = []

    opt = optimizer
    sch = scheduler

    def _rebuild_optimizer_like(old_opt: torch.optim.Optimizer, new_model: torch.nn.Module) -> torch.optim.Optimizer:
        if isinstance(old_opt, torch.optim.Adam):
            return torch.optim.Adam(new_model.parameters(), lr=old_opt.param_groups[0]["lr"])
        if isinstance(old_opt, torch.optim.AdamW):
            return torch.optim.AdamW(new_model.parameters(), lr=old_opt.param_groups[0]["lr"])
        if isinstance(old_opt, torch.optim.RMSprop):
            return torch.optim.RMSprop(new_model.parameters(), lr=old_opt.param_groups[0]["lr"])
        raise ValueError(f"Unsupported optimizer type for rebuild: {type(old_opt)}")

    def _rebuild_scheduler_like(old_sch, new_opt: torch.optim.Optimizer, cfg):
        if old_sch is None:
            return None
        if isinstance(old_sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                new_opt,
                **getattr(cfg, "schedulerparams", {}),
            )
        raise ValueError(f"Unsupported scheduler type for rebuild: {type(old_sch)}")

    def _record(avg_data: float, avg_imag_w_reg: float):
        data_losses.append(float(avg_data))
        imag_w_losses.append(float(avg_imag_w_reg))

    def _maybe_print(
        tag: str,
        avg_total: float,
        avg_data: float,
        avg_sparse: float,
        avg_imag_w: float,
        avg_theta: float,
        valid_frac: float,
    ):
        if (global_epoch + 1) % cfg.print_every == 0 or global_epoch == 0:
            lr = opt.param_groups[0]["lr"]
            active = model.count_active_edges()
            print(
                f"[{tag} | Epoch {global_epoch+1}] "
                f"lr={lr:.2e}, total={avg_total:.4e}, data={avg_data:.4e}, "
                f"sparsity_reg={avg_sparse:.4e}, imag_w={avg_imag_w:.4e}, "
                f"theta={avg_theta:.4e}, valid={valid_frac:.3f}, "
                f"active_edges={active}"
            )
            if on_print is not None:
                on_print(global_epoch + 1, model)

    def _get_one_batch():
        xb, yb = next(iter(dataloader))
        return xb.to(device), yb.to(device)

    def _do_prune_and_rebuild(tag: str, prune_fraction: float):
        nonlocal model, opt, sch

        if bool(getattr(cfg, "ablation_prune_enabled", False)) and prune_fraction > 0.0:
            xb, yb = _get_one_batch()
            before_edges = model.count_active_edges()

            pruned_abl = ablation_prune_on_batch_wrapper(
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
            

            # dropped_invalid = 0
            # xb_aux = xb[:, 1:]

            # for ridx, r in enumerate(model.resonators):
            #     dropped_here = r.eql.drop_invalid_ops_on_batch_(
            #         xb_aux,
            #         op_params=None,
            #         verbose=True,
            #     )
            #     dropped_invalid += dropped_here
            #     if dropped_here > 0:
            #         print(f"[DROP_INVALID_OPS | {tag}] resonator={ridx} dropped_ops={dropped_here}")

            # if dropped_invalid > 0:
            #     print(f"[DROP_INVALID_OPS | {tag}] total_dropped_ops={dropped_invalid}")


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
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, avg_theta, _, valid_frac = train_one_epoch_wrapper(
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
            normalize_divisions=False,
            normalize_divisions_eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)),
            imag_shrink_enabled=bool(getattr(cfg, "phase1_imag_shrink_enabled", False)),
            imag_shrink_coeff=float(getattr(cfg, "phase1_imag_shrink_coeff", 1.0)),
            theta_penalty_enabled=True,
            theta_penalty_coeff=float(getattr(cfg, "theta_coeff_phase1", 0.0)),
            theta_penalty_eps=float(getattr(cfg, "theta_eps", 1e-12)),
        )
        _maybe_print("PHASE1", avg_total, avg_data, avg_sparse, avg_imag_w, avg_theta, valid_frac)
        _record(avg_data, avg_imag_w)
        global_epoch += 1

    # -------------------------
    # End Phase 1
    # -------------------------
    phase1_prune_enabled = bool(getattr(cfg, "phase1_prune_enabled", True))
    phase1_prune_thr = float(getattr(cfg, "phase1_prune_threshold", 0.0))

    if phase1_prune_enabled and phase1_prune_thr > 0.0:
        tag = "PHASE1_END"

        model.normalize_all_divisions_(eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)))
        before = model.count_active_edges()

        pruned = prune_by_threshold_wrapper(model, phase1_prune_thr)

        dropped_div = model.drop_div_ops_with_pruned_denominator_()
        dropped_log = model.drop_log_ops_with_pruned_input_()
        cleaned = model.cascade_cleanup_disconnected_()

        if dropped_div > 0:
            print(f"[DROP_DIV | {tag}] pruned_downstream_edges={dropped_div}")
        if dropped_log > 0:
            print(f"[DROP_LOG | {tag}] pruned_downstream_edges={dropped_log}")
        if cleaned > 0:
            print(f"[CLEAN | {tag}] cleaned_disconnected={cleaned}")

        before_rebuild = model.count_active_edges()
        model = model.rebuild_from_pruned().to(device)
        opt = _rebuild_optimizer_like(opt, model)
        sch = _rebuild_scheduler_like(sch, opt, cfg)
        after = model.count_active_edges()

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
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, avg_theta, _, valid_frac = train_one_epoch_wrapper(
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
            normalize_divisions=normalize_divs,
            normalize_divisions_eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)),
            imag_shrink_enabled=bool(getattr(cfg, "phase2_imag_shrink_enabled", False)),
            imag_shrink_coeff=float(getattr(cfg, "phase2_imag_shrink_coeff", 1.0)),
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
        model.normalize_all_divisions_(eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)))
        model.force_real_()
        model.freeze_imag_()

    for _ in range(phase3_epochs):
        avg_total, avg_data, _avg_reg, avg_sparse, avg_imag_w, avg_theta, _, valid_frac = train_one_epoch_wrapper(
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
            normalize_divisions=normalize_divs,
            normalize_divisions_eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)),
            imag_shrink_enabled=bool(getattr(cfg, "phase3_imag_shrink_enabled", False)),
            imag_shrink_coeff=float(getattr(cfg, "phase3_imag_shrink_coeff", 1.0)),
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
        model.normalize_all_divisions_(eps=float(getattr(cfg, "normalize_divisions_eps", 1e-12)))
        model.force_real_()
        model.freeze_imag_()

    return model, (imag_w_losses, data_losses)