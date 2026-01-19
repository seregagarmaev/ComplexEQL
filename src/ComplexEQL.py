from __future__ import annotations

from typing import Any, List, Optional
import numpy as np
import sympy as sp
import torch
import torch.nn as nn

from src.operations import OpParams, load_models
from src.sympy_utils import prune_small_coeff_terms
from src.operations import nan_to_num_complex, clamp_complex, CLAMP_VAL


def _threshold_multiply(coef: sp.Number, val: sp.Expr, decimals: int) -> sp.Expr:
    c = sp.N(coef, max(16, decimals + 6))
    if c.is_number is True:
        thr = 0.5 * (10.0 ** (-int(decimals)))
        if abs(float(c)) < thr:
            return sp.Integer(0)
        return sp.N(coef * val, decimals)
    return coef * val


def nan_to_num_complex(
    x: torch.Tensor, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0
) -> torch.Tensor:
    if torch.is_complex(x):
        return torch.complex(
            torch.nan_to_num(x.real, nan=nan, posinf=posinf, neginf=neginf),
            torch.nan_to_num(x.imag, nan=nan, posinf=posinf, neginf=neginf),
        )
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


_BAD_SYMPY_ATOMS = (sp.oo, -sp.oo, sp.zoo, sp.nan)


def _sanitize_symbolic(expr: Any) -> sp.Expr:
    if expr is None:
        return sp.Integer(0)
    if isinstance(expr, bool):
        return sp.Integer(int(expr))
    if isinstance(expr, int):
        return sp.Integer(expr)
    if isinstance(expr, float):
        return sp.Float(expr)
    if not isinstance(expr, sp.Basic):
        return sp.Integer(0)

    if expr.has(*_BAD_SYMPY_ATOMS):
        return sp.Integer(0)
    if getattr(expr, "is_infinite", None) is True:
        return sp.Integer(0)
    if getattr(expr, "is_nan", None) is True:
        return sp.Integer(0)

    return expr


def _is_exact_zero(e: sp.Expr) -> bool:
    if not isinstance(e, sp.Basic):
        return False
    if e.is_zero is True:
        return True
    if e.is_number is True and e == 0:
        return True
    return False


class SymbolicLayer(nn.Module):
    def __init__(self, cfg, layer_number: int, n_input_fields: int, *, op_specs=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_number = int(layer_number)
        self.n_input_fields = int(n_input_fields)

        unary_ops, binary_ops = load_models(cfg, self.layer_number, specs_override=op_specs)
        self.unary_ops = nn.ModuleList(unary_ops)
        self.binary_ops = nn.ModuleList(binary_ops)

        self.n_unary_ops = len(self.unary_ops)
        self.n_binary_ops = len(self.binary_ops)

        self.n_ops = self.n_unary_ops + self.n_binary_ops
        self.n_inputs = self.n_unary_ops + 2 * self.n_binary_ops

        scale = 0.1 ** (3 - self.layer_number)
        real = (torch.rand(self.n_input_fields, self.n_inputs) - 0.5) * scale
        imag = (torch.rand(self.n_input_fields, self.n_inputs) - 0.5) * scale
        self.weights = nn.Parameter(torch.complex(real, imag))
        self.mask = nn.Parameter(torch.ones_like(real), requires_grad=False)

        self.function_names = [m.fname for m in self.unary_ops] + [m.fname for m in self.binary_ops]
        self.functions_dict = cfg.functions_dict

    def get_symbolic_output(
        self,
        symbolic_inputs: List[sp.Expr],
        rounding_decimals: int = 2,
        *,
        use_imag: bool = False,
    ) -> List[sp.Expr]:
        outs: List[sp.Expr] = []

        for op_idx in range(self.n_unary_ops):
            op_name = self.function_names[op_idx]
            mixed: sp.Expr = sp.Integer(0)

            for j in range(len(symbolic_inputs)):
                w = self.weights[j, op_idx].detach()
                coef_r = sp.Float(float(w.real.cpu().item()))
                mixed += _threshold_multiply(coef_r, symbolic_inputs[j], rounding_decimals)

                if use_imag:
                    coef_i = sp.Float(float(w.imag.cpu().item()))
                    mixed += sp.I * _threshold_multiply(coef_i, symbolic_inputs[j], rounding_decimals)

            op = self.functions_dict[op_name]

            if op_name == "log":
                symbolic_out = sp.Integer(0) if _is_exact_zero(mixed) else op(mixed)
            elif op_name == "sqrt":
                symbolic_out = sp.Integer(0) if _is_exact_zero(mixed) else op(mixed)
            else:
                symbolic_out = op(mixed)

            outs.append(_sanitize_symbolic(symbolic_out))

        for i in range(self.n_binary_ops):
            op_name = self.function_names[self.n_unary_ops + i]
            op = self.functions_dict[op_name]

            a: sp.Expr = sp.Integer(0)
            b: sp.Expr = sp.Integer(0)
            a_col = self.n_unary_ops + 2 * i
            b_col = a_col + 1

            for j in range(len(symbolic_inputs)):
                w_a = self.weights[j, a_col].detach()
                w_b = self.weights[j, b_col].detach()

                a += _threshold_multiply(sp.Float(float(w_a.real.cpu().item())), symbolic_inputs[j], rounding_decimals)
                b += _threshold_multiply(sp.Float(float(w_b.real.cpu().item())), symbolic_inputs[j], rounding_decimals)

                if use_imag:
                    a += sp.I * _threshold_multiply(
                        sp.Float(float(w_a.imag.cpu().item())), symbolic_inputs[j], rounding_decimals
                    )
                    b += sp.I * _threshold_multiply(
                        sp.Float(float(w_b.imag.cpu().item())), symbolic_inputs[j], rounding_decimals
                    )

            if op_name == "div":
                expr = sp.Integer(0) if _is_exact_zero(b) else (a / b)
                outs.append(_sanitize_symbolic(expr))
            else:
                outs.append(_sanitize_symbolic(op(a, b)))

        return outs

    @torch.no_grad()
    def normalize_division_mixing_(self, eps: float = 1e-12) -> int:
        normalized = 0
        if self.n_binary_ops == 0:
            return 0

        for i in range(self.n_binary_ops):
            op_name = self.function_names[self.n_unary_ops + i]
            if op_name != "div":
                continue

            a_col = self.n_unary_ops + 2 * i
            b_col = a_col + 1

            m_a = self.mask[:, a_col].bool()
            m_b = self.mask[:, b_col].bool()
            if (not m_a.any()) and (not m_b.any()):
                continue

            w_a = self.weights[:, a_col]
            w_b = self.weights[:, b_col]

            max_a = w_a[m_a].abs().max() if m_a.any() else torch.tensor(0.0, device=w_a.device)
            max_b = w_b[m_b].abs().max() if m_b.any() else torch.tensor(0.0, device=w_b.device)
            scale = torch.maximum(max_a, max_b)

            if scale <= eps:
                continue

            self.weights[:, a_col] = torch.where(m_a, w_a / scale, w_a)
            self.weights[:, b_col] = torch.where(m_b, w_b / scale, w_b)

            self.weights.data = nan_to_num_complex(self.weights.data, nan=0.0, posinf=0.0, neginf=0.0)
            normalized += 1

        return normalized

    def weights_for_reg(self) -> torch.Tensor:
        effective = self.weights * self.mask.to(self.weights.dtype)
        return effective.reshape(-1)

    def lift(self, X: torch.Tensor) -> torch.Tensor:
        effective = self.weights * self.mask.to(self.weights.dtype)
        return torch.matmul(X.to(effective.dtype), effective)

    def prune_by_threshold(self, threshold: float) -> int:
        with torch.no_grad():
            active = (self.mask == 1.0)

            non_finite = (~torch.isfinite(self.weights.real)) | (~torch.isfinite(self.weights.imag))
            small = (self.weights.abs() < threshold)

            to_prune = active & (non_finite | small)
            num_pruned = int(to_prune.sum().item())

            self.mask[to_prune] = 0.0
            self.weights[to_prune] = torch.complex(
                torch.zeros_like(self.weights.real[to_prune]),
                torch.zeros_like(self.weights.imag[to_prune]),
            )
            self.weights.data = nan_to_num_complex(self.weights.data, nan=0.0, posinf=0.0, neginf=0.0)

        return num_pruned

    def apply_operations(self, X: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
        results: List[torch.Tensor] = []

        # unary: contiguous slice per op
        for i, op in enumerate(self.unary_ops):
            results.append(op(X[:, i], op_params=op_params))

        # binary: slice two columns, no stack
        for i, op in enumerate(self.binary_ops):
            start = self.n_unary_ops + 2 * i
            results.append(op(X[:, start], X[:, start + 1], op_params=op_params))

        return torch.cat(results, dim=-1)

    # def forward(self, X: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
    #     lifted = self.lift(X)
    #     return self.apply_operations(lifted, op_params=op_params)
    def forward(self, X: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
        lifted = self.lift(X)
        out = self.apply_operations(lifted, op_params=op_params)

        out = nan_to_num_complex(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = clamp_complex(out, -CLAMP_VAL, CLAMP_VAL)
        return out


class AssemblyLayer(nn.Module):
    def __init__(self, cfg, in_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = int(in_dim)

        real = (torch.rand(in_dim, 1) - 0.5)
        imag = (torch.rand(in_dim, 1) - 0.5)
        self.weights = nn.Parameter(torch.complex(real, imag))
        self.mask = nn.Parameter(torch.ones_like(real), requires_grad=False)

    def get_symbolic_output(
        self,
        symbolic_inputs: List[sp.Expr],
        rounding_decimals: int = 2,
        *,
        use_imag: bool = False,
    ) -> sp.Expr:
        out: sp.Expr = sp.Integer(0)
        for i in range(len(symbolic_inputs)):
            w = self.weights[i, 0].detach()
            coef_r = sp.Float(float(w.real.cpu().item()))
            out += _threshold_multiply(coef_r, symbolic_inputs[i], rounding_decimals)

            if use_imag:
                coef_i = sp.Float(float(w.imag.cpu().item()))
                out += sp.I * _threshold_multiply(coef_i, symbolic_inputs[i], rounding_decimals)

        return _sanitize_symbolic(out)

    def prune_by_threshold(self, threshold: float) -> int:
        with torch.no_grad():
            active = (self.mask == 1.0)

            non_finite = (~torch.isfinite(self.weights.real)) | (~torch.isfinite(self.weights.imag))
            small = (self.weights.abs() < threshold)

            to_prune = active & (non_finite | small)
            num_pruned = int(to_prune.sum().item())

            self.mask[to_prune] = 0.0
            self.weights[to_prune] = torch.complex(
                torch.zeros_like(self.weights.real[to_prune]),
                torch.zeros_like(self.weights.imag[to_prune]),
            )
            self.weights.data = nan_to_num_complex(self.weights.data, nan=0.0, posinf=0.0, neginf=0.0)

        return num_pruned

    def weights_for_reg(self) -> torch.Tensor:
        effective = self.weights * self.mask.to(self.weights.dtype)
        return effective.reshape(-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        effective = self.weights * self.mask.to(self.weights.dtype)
        return torch.matmul(X, effective)


class ComplexEQL(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        layers: list[SymbolicLayer] = []
        prev_n_ops: int | None = None

        for layer_idx in range(cfg.n_symbolic_layers):
            n_in = int(cfg.n_input_fields) if layer_idx == 0 else int(cfg.n_input_fields) + int(prev_n_ops)
            layers.append(SymbolicLayer(cfg, layer_idx, n_input_fields=n_in).to(cfg.device))
            prev_n_ops = layers[-1].n_ops

        self.symbolic_layers = nn.ModuleList(layers)
        last_outputs = self.symbolic_layers[-1].n_ops
        self.assembly_layer = AssemblyLayer(cfg, int(cfg.n_input_fields) + int(last_outputs))

    def get_symbolic_expression(
        self,
        symbolic_inputs: List[sp.Expr],
        rounding_decimals: int = 2,
        *,
        use_imag: bool = False,
    ) -> sp.Expr:
        sym_x0: List[sp.Expr] = symbolic_inputs
        sym_h: List[sp.Expr] = self.symbolic_layers[0].get_symbolic_output(
            sym_x0, rounding_decimals=rounding_decimals, use_imag=use_imag
        )

        for layer in self.symbolic_layers[1:]:
            sym_h = layer.get_symbolic_output(
                sym_x0 + sym_h, rounding_decimals=rounding_decimals, use_imag=use_imag
            )

        sym_out = self.assembly_layer.get_symbolic_output(
            sym_x0 + sym_h, rounding_decimals=rounding_decimals, use_imag=use_imag
        )
        sym_out = _sanitize_symbolic(sym_out)
        return prune_small_coeff_terms(sym_out, rounding_decimals)

    def get_real_weights_list(self) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for layer in self.symbolic_layers:
            eff = layer.weights * layer.mask.to(layer.weights.dtype)
            out.append(eff.real.view(-1))
        effA = self.assembly_layer.weights * self.assembly_layer.mask.to(self.assembly_layer.weights.dtype)
        out.append(effA.real.view(-1))
        return out

    def get_imag_weights_list(self) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for layer in self.symbolic_layers:
            eff = layer.weights * layer.mask.to(layer.weights.dtype)
            out.append(eff.imag.view(-1))
        effA = self.assembly_layer.weights * self.assembly_layer.mask.to(self.assembly_layer.weights.dtype)
        out.append(effA.imag.view(-1))
        return out

    @torch.no_grad()
    def count_active_edges(self) -> int:
        total = 0
        for layer in self.symbolic_layers:
            total += int((layer.mask > 0.5).sum().item())
        total += int((self.assembly_layer.mask > 0.5).sum().item())
        return total

    @torch.no_grad()
    def count_active_edges_per_layer(self) -> tuple[list[int], int]:
        per_sym = [int((layer.mask > 0.5).sum().item()) for layer in self.symbolic_layers]
        asm = int((self.assembly_layer.mask > 0.5).sum().item())
        return per_sym, asm

    @torch.no_grad()
    def pruning_threshold_from_fraction(
        self,
        fraction: float,
        *,
        min_edges_total: int,
        eps: float = 1e-12,
    ) -> tuple[float | None, int, int]:
        fraction = float(fraction)
        if fraction <= 0.0:
            active = self.count_active_edges()
            return (None, 0, active)

        active = self.count_active_edges()
        if active <= int(min_edges_total):
            return (None, 0, active)

        mags_list: list[torch.Tensor] = []
        for layer in self.symbolic_layers:
            m = (layer.mask > 0.5)
            if m.any():
                mags_list.append(layer.weights.abs()[m].reshape(-1))
        mA = (self.assembly_layer.mask > 0.5)
        if mA.any():
            mags_list.append(self.assembly_layer.weights.abs()[mA].reshape(-1))

        if not mags_list:
            return (None, 0, active)

        mags = torch.cat(mags_list, dim=0)
        n = int(mags.numel())

        k_target = int(np.floor(fraction * n))
        k_cap = n - int(min_edges_total)
        k_to_prune = max(0, min(k_target, k_cap))
        if k_to_prune <= 0:
            return (None, 0, active)

        kth = torch.kthvalue(mags, k_to_prune).values
        inf = torch.tensor(float("inf"), device=kth.device, dtype=kth.dtype)
        thr = torch.nextafter(kth, inf)
        thr_val = float(max(float(thr.item()), float(eps)))
        return (thr_val, k_to_prune, active)

    @torch.no_grad()
    def pruning_thresholds_from_fraction_per_layer(
        self,
        fraction: float,
        *,
        min_edges_per_layer: int,
        eps: float = 1e-12,
    ) -> tuple[list[tuple[float | None, int, int]], tuple[float | None, int, int]]:
        fraction = float(fraction)
        min_edges_per_layer = int(min_edges_per_layer)

        def _layer_threshold(mags: torch.Tensor, active_now: int) -> tuple[float | None, int, int]:
            if fraction <= 0.0 or active_now <= min_edges_per_layer:
                return (None, 0, active_now)

            n = int(mags.numel())
            if n == 0:
                return (None, 0, active_now)

            k_target = int(np.floor(fraction * n))
            k_cap = n - min_edges_per_layer
            k_to_prune = max(0, min(k_target, k_cap))
            if k_to_prune <= 0:
                return (None, 0, active_now)

            kth = torch.kthvalue(mags, k_to_prune).values
            inf = torch.tensor(float("inf"), device=kth.device, dtype=kth.dtype)
            thr = torch.nextafter(kth, inf)
            thr_val = float(max(float(thr.item()), float(eps)))
            return (thr_val, k_to_prune, active_now)

        per_layer: list[tuple[float | None, int, int]] = []
        for layer in self.symbolic_layers:
            m = (layer.mask > 0.5)
            active_now = int(m.sum().item())
            mags = (
                layer.weights.abs()[m].reshape(-1)
                if active_now > 0
                else torch.empty(0, device=layer.weights.device)
            )
            per_layer.append(_layer_threshold(mags, active_now))

        mA = (self.assembly_layer.mask > 0.5)
        activeA = int(mA.sum().item())
        magsA = (
            self.assembly_layer.weights.abs()[mA].reshape(-1)
            if activeA > 0
            else torch.empty(0, device=self.assembly_layer.weights.device)
        )
        asm_tuple = _layer_threshold(magsA, activeA)

        return per_layer, asm_tuple

    def prune_by_threshold(self, threshold: float) -> int:
        total = 0
        for layer in self.symbolic_layers:
            total += layer.prune_by_threshold(threshold)
        total += self.assembly_layer.prune_by_threshold(threshold)
        return total

    @torch.no_grad()
    def cascade_cleanup_disconnected_(self) -> int:
        n0 = int(self.cfg.n_input_fields)
        L = len(self.symbolic_layers)
        newly_pruned = 0

        asm_m = (self.assembly_layer.mask[:, 0] > 0.5)
        last_ops = self.symbolic_layers[-1].n_ops
        req_h_next = asm_m[n0 : n0 + last_ops].clone()

        for layer_idx in range(L - 1, -1, -1):
            layer = self.symbolic_layers[layer_idx]
            req_out = req_h_next.clone()

            for k in range(layer.n_unary_ops):
                if not bool(req_out[k].item()):
                    col = k
                    active_mask = (layer.mask[:, col] > 0.5)
                    if active_mask.any():
                        newly_pruned += int(active_mask.sum().item())
                        layer.mask[active_mask, col] = 0.0
                        layer.weights[active_mask, col] = torch.complex(
                            torch.zeros_like(layer.weights.real[active_mask, col]),
                            torch.zeros_like(layer.weights.imag[active_mask, col]),
                        )

            for i in range(layer.n_binary_ops):
                out_idx = layer.n_unary_ops + i
                if not bool(req_out[out_idx].item()):
                    a_col = layer.n_unary_ops + 2 * i
                    b_col = a_col + 1

                    active_a = (layer.mask[:, a_col] > 0.5)
                    active_b = (layer.mask[:, b_col] > 0.5)

                    if active_a.any():
                        newly_pruned += int(active_a.sum().item())
                        layer.mask[active_a, a_col] = 0.0
                        layer.weights[active_a, a_col] = torch.complex(
                            torch.zeros_like(layer.weights.real[active_a, a_col]),
                            torch.zeros_like(layer.weights.imag[active_a, a_col]),
                        )
                    if active_b.any():
                        newly_pruned += int(active_b.sum().item())
                        layer.mask[active_b, b_col] = 0.0
                        layer.weights[active_b, b_col] = torch.complex(
                            torch.zeros_like(layer.weights.real[active_b, b_col]),
                            torch.zeros_like(layer.weights.imag[active_b, b_col]),
                        )

            req_in_this_layer = torch.zeros(layer.n_input_fields, dtype=torch.bool, device=layer.weights.device)

            for k in range(layer.n_unary_ops):
                if not bool(req_out[k].item()):
                    continue
                col = k
                req_in_this_layer |= (layer.mask[:, col] > 0.5)

            for i in range(layer.n_binary_ops):
                out_idx = layer.n_unary_ops + i
                if not bool(req_out[out_idx].item()):
                    continue
                a_col = layer.n_unary_ops + 2 * i
                b_col = a_col + 1
                req_in_this_layer |= (layer.mask[:, a_col] > 0.5) | (layer.mask[:, b_col] > 0.5)

            layer.weights.data = nan_to_num_complex(layer.weights.data, nan=0.0, posinf=0.0, neginf=0.0)

            if layer_idx == 0:
                break
            req_h_next = req_in_this_layer[n0:]

        return newly_pruned

    @torch.no_grad()
    def _required_outputs_per_layer(self) -> list[torch.Tensor]:
        n0 = int(self.cfg.n_input_fields)
        L = len(self.symbolic_layers)

        req_outs: list[torch.Tensor] = [None] * L

        asm_m = (self.assembly_layer.mask[:, 0] > 0.5)
        last_ops = self.symbolic_layers[-1].n_ops
        req_h_next = asm_m[n0 : n0 + last_ops].clone()
        req_outs[L - 1] = req_h_next.clone()

        for layer_idx in range(L - 1, 0, -1):
            layer = self.symbolic_layers[layer_idx]
            req_out = req_outs[layer_idx]

            req_in = torch.zeros(layer.n_input_fields, dtype=torch.bool, device=layer.weights.device)

            for k in range(layer.n_unary_ops):
                if bool(req_out[k].item()):
                    req_in |= (layer.mask[:, k] > 0.5)

            for i in range(layer.n_binary_ops):
                out_idx = layer.n_unary_ops + i
                if bool(req_out[out_idx].item()):
                    a_col = layer.n_unary_ops + 2 * i
                    b_col = a_col + 1
                    req_in |= (layer.mask[:, a_col] > 0.5) | (layer.mask[:, b_col] > 0.5)

            req_outs[layer_idx - 1] = req_in[n0:].clone()

        return req_outs

    @torch.no_grad()
    def rebuild_from_pruned(self) -> "ComplexEQL":
        cfg = self.cfg
        device = self.symbolic_layers[0].weights.device
        dtype = self.symbolic_layers[0].weights.dtype
        n0 = int(cfg.n_input_fields)

        req_outs = self._required_outputs_per_layer()

        kept_out_indices_per_layer: list[list[int]] = []
        for li, layer in enumerate(self.symbolic_layers):
            req = req_outs[li]
            kept = [j for j in range(layer.n_ops) if bool(req[j].item())]
            kept_out_indices_per_layer.append(kept)

        # Drop trailing layers that keep no ops (these would create empty SymbolicLayer -> torch.cat([])).
        last_nonempty = -1
        for li, kept in enumerate(kept_out_indices_per_layer):
            if len(kept) > 0:
                last_nonempty = li

        # If everything is empty, keep the first layer as a fallback (do NOT allow 0-layer model here).
        # This preserves your existing forward/get_symbolic_expression assumptions.
        if last_nonempty < 0:
            last_nonempty = 0

        # We will rebuild only up to last_nonempty (inclusive).
        kept_out_indices_per_layer = kept_out_indices_per_layer[: last_nonempty + 1]
        old_layers_to_rebuild = list(self.symbolic_layers[: last_nonempty + 1])

        # NEW: ensure no rebuilt layer is empty; otherwise apply_operations() would torch.cat([]).
        for li, kept in enumerate(kept_out_indices_per_layer):
            if len(kept) == 0:
                kept_out_indices_per_layer[li] = list(range(old_layers_to_rebuild[li].n_ops))

        new_layers: list[SymbolicLayer] = []
        prev_kept_outputs_old: list[int] = []

        for li, old_layer in enumerate(old_layers_to_rebuild):
            keep_rows = list(range(n0)) + [n0 + j for j in prev_kept_outputs_old]

            kept_out = kept_out_indices_per_layer[li]
            kept_unary = [k for k in kept_out if k < old_layer.n_unary_ops]
            kept_binary_out = [k for k in kept_out if k >= old_layer.n_unary_ops]
            kept_binary = [k - old_layer.n_unary_ops for k in kept_binary_out]

            op_specs: list[dict] = []
            for k in kept_unary:
                op_specs.append({"op": old_layer.unary_ops[k].fname, "type": "unary"})
            for i in kept_binary:
                op_specs.append({"op": old_layer.binary_ops[i].fname, "type": "binary"})

            keep_cols: list[int] = []
            for k in kept_unary:
                keep_cols.append(k)
            for i in kept_binary:
                a_col = old_layer.n_unary_ops + 2 * i
                keep_cols.append(a_col)
                keep_cols.append(a_col + 1)

            n_input_fields_new = len(keep_rows)
            new_layer = SymbolicLayer(cfg, li, n_input_fields=n_input_fields_new, op_specs=op_specs).to(device)

            w_new = old_layer.weights.data[keep_rows][:, keep_cols].to(device=device, dtype=dtype)
            m_new = old_layer.mask.data[keep_rows][:, keep_cols].to(device=device, dtype=old_layer.mask.dtype)

            new_layer.weights.data.copy_(w_new)
            new_layer.mask.data.copy_(m_new)

            new_layers.append(new_layer)
            prev_kept_outputs_old = kept_out

        old_last = old_layers_to_rebuild[-1]
        last_kept_out = kept_out_indices_per_layer[-1]
        asm_keep_rows = list(range(n0)) + [n0 + j for j in last_kept_out]

        new_model = ComplexEQL.__new__(ComplexEQL)
        nn.Module.__init__(new_model)
        new_model.cfg = cfg
        new_model.symbolic_layers = nn.ModuleList(new_layers)

        new_in_dim = n0 + len(last_kept_out)
        new_model.assembly_layer = AssemblyLayer(cfg, new_in_dim).to(device)

        wA_new = self.assembly_layer.weights.data[asm_keep_rows].to(
            device=device, dtype=self.assembly_layer.weights.dtype
        )
        mA_new = self.assembly_layer.mask.data[asm_keep_rows].to(
            device=device, dtype=self.assembly_layer.mask.dtype
        )

        new_model.assembly_layer.weights.data.copy_(wA_new)
        new_model.assembly_layer.mask.data.copy_(mA_new)

        return new_model

    @torch.no_grad()
    def normalize_all_divisions_(self, eps: float = 1e-12) -> int:
        total = 0
        for layer in self.symbolic_layers:
            total += layer.normalize_division_mixing_(eps=eps)
        return total

    def sanitize_gradients(self, max_grad: float = 1e3) -> None:
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is None:
                    continue
                g = p.grad.data
                g = nan_to_num_complex(g, nan=0.0, posinf=0.0, neginf=0.0)
                mag = torch.abs(g)
                mask = mag > max_grad
                if mask.any():
                    scale = max_grad / mag[mask]
                    g = g.clone()
                    g[mask] = g[mask] * scale
                p.grad.data.copy_(g)

    def sanitize_weights(self, clamp_value: float = 1e6) -> None:
        def _clamp_complex_by_magnitude(x: torch.Tensor, max_mag: float) -> torch.Tensor:
            mag = torch.abs(x)
            mask = mag > max_mag
            if mask.any():
                scale = max_mag / mag[mask]
                x = x.clone()
                x[mask] = x[mask] * scale
            return x

        with torch.no_grad():
            for layer in self.symbolic_layers:
                w = layer.weights.data
                w = nan_to_num_complex(w, nan=0.0, posinf=0.0, neginf=0.0)
                w = _clamp_complex_by_magnitude(w, clamp_value)
                layer.weights.data.copy_(w)

            w = self.assembly_layer.weights.data
            w = nan_to_num_complex(w, nan=0.0, posinf=0.0, neginf=0.0)
            w = _clamp_complex_by_magnitude(w, clamp_value)
            self.assembly_layer.weights.data.copy_(w)

    @torch.no_grad()
    def shrink_imag_weights_(self, coeff: float) -> None:
        coeff = float(coeff)
        if coeff >= 1.0:
            return
        if coeff < 0.0:
            raise ValueError("shrink coeff must be in [0,1].")

        for layer in self.symbolic_layers:
            w = layer.weights.data
            layer.weights.data = torch.complex(w.real, w.imag * coeff)

        w = self.assembly_layer.weights.data
        self.assembly_layer.weights.data = torch.complex(w.real, w.imag * coeff)

    def forward(self, x: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
        x0 = x
        h = self.symbolic_layers[0](x0, op_params=op_params)

        x0_c = x0.to(h.dtype)
        for layer in self.symbolic_layers[1:]:
            h = layer(torch.cat([x0_c, h], dim=1), op_params=op_params)

        y = self.assembly_layer(torch.cat([x0_c, h], dim=1))
        return y
