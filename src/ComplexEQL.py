from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import sympy
import torch
import torch.nn as nn

from src.operations import OpParams, load_models
from src.sympy_utils import prune_small_coeff_terms


def _threshold_multiply(coef: sympy.Number, val: sympy.Expr, decimals: int) -> sympy.Expr:
    try:
        if coef.round(decimals) == 0.0:
            return sympy.Integer(0)
        return (coef * val).n(decimals)
    except Exception:
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


class SymbolicLayer(nn.Module):
    def __init__(self, cfg, layer_number: int, n_input_fields: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_number = layer_number
        self.n_input_fields = int(n_input_fields)

        unary_ops, binary_ops = load_models(cfg, layer_number)
        self.unary_ops = nn.ModuleList(unary_ops)
        self.binary_ops = nn.ModuleList(binary_ops)

        self.n_unary_ops = len(self.unary_ops)
        self.n_binary_ops = len(self.binary_ops)

        self.n_ops = self.n_unary_ops + self.n_binary_ops
        self.n_inputs = self.n_unary_ops + 2 * self.n_binary_ops

        real = (torch.rand(self.n_input_fields, self.n_inputs) - 0.5) * 0.1
        imag = (torch.rand(self.n_input_fields, self.n_inputs) - 0.5) * 0.1
        self.weights = nn.Parameter(torch.complex(real, imag))
        self.mask = nn.Parameter(torch.ones_like(real), requires_grad=False)

        self.function_names = [m.fname for m in self.unary_ops] + [m.fname for m in self.binary_ops]
        self.functions_dict = cfg.functions_dict

    def get_symbolic_output(
        self,
        symbolic_inputs: List[sympy.Expr],
        rounding_decimals: int = 2,
    ) -> List[sympy.Expr]:
        outs: List[sympy.Expr] = []

        # unary: columns [0 .. n_unary_ops-1]
        for op_idx in range(self.n_unary_ops):
            op_name = self.function_names[op_idx]
            mixed = sympy.Integer(0)

            for j in range(len(symbolic_inputs)):
                w = self.weights[j, op_idx].detach()
                coef_r = sympy.Float(float(w.real.cpu().item()))
                coef_i = sympy.Float(float(w.imag.cpu().item()))

                mixed += _threshold_multiply(coef_r, symbolic_inputs[j], rounding_decimals)
                mixed += sympy.I * _threshold_multiply(coef_i, symbolic_inputs[j], rounding_decimals)

            op = self.functions_dict[op_name]
            symbolic_out = op(mixed)

            if isinstance(symbolic_out, int) or getattr(symbolic_out, "is_infinite", False):
                outs.append(sympy.Integer(0))
            else:
                outs.append(sympy.sympify(symbolic_out))

        # binary: each op uses two columns
        for i in range(self.n_binary_ops):
            op_name = self.function_names[self.n_unary_ops + i]
            op = self.functions_dict[op_name]

            a = sympy.Integer(0)
            b = sympy.Integer(0)
            a_col = self.n_unary_ops + 2 * i
            b_col = a_col + 1

            for j in range(len(symbolic_inputs)):
                w_a = self.weights[j, a_col].detach()
                w_b = self.weights[j, b_col].detach()

                a += _threshold_multiply(sympy.Float(float(w_a.real.cpu().item())), symbolic_inputs[j], rounding_decimals)
                a += sympy.I * _threshold_multiply(sympy.Float(float(w_a.imag.cpu().item())), symbolic_inputs[j], rounding_decimals)

                b += _threshold_multiply(sympy.Float(float(w_b.real.cpu().item())), symbolic_inputs[j], rounding_decimals)
                b += sympy.I * _threshold_multiply(sympy.Float(float(w_b.imag.cpu().item())), symbolic_inputs[j], rounding_decimals)

            if op_name == "div":
                try:
                    outs.append(sympy.cancel(a / b))
                except Exception:
                    outs.append(sympy.sympify(op(a, b)))
            else:
                outs.append(sympy.sympify(op(a, b)))

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
            active = self.mask == 1.0
            to_prune = active & (self.weights.abs() < threshold)
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

        for i, op in enumerate(self.unary_ops):
            results.append(op(X[:, i], op_params=op_params))

        for i in range(self.n_binary_ops):
            start = self.n_unary_ops + 2 * i
            pair = torch.stack((X[:, start], X[:, start + 1]), dim=1)
            results.append(self.binary_ops[i](pair, op_params=op_params))

        return torch.cat(results, dim=-1)

    def forward(self, X: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
        lifted = self.lift(X)
        return self.apply_operations(lifted, op_params=op_params)


class AssemblyLayer(nn.Module):
    def __init__(self, cfg, in_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = int(in_dim)

        real = (torch.rand(in_dim, 1) - 0.5) * 0.1
        imag = (torch.rand(in_dim, 1) - 0.5) * 0.1
        self.weights = nn.Parameter(torch.complex(real, imag))
        self.mask = nn.Parameter(torch.ones_like(real), requires_grad=False)

    def get_symbolic_output(
        self,
        symbolic_inputs: List[sympy.Expr],
        rounding_decimals: int = 2,
    ) -> sympy.Expr:
        out: sympy.Expr = sympy.Integer(0)
        for i in range(len(symbolic_inputs)):
            w = self.weights[i, 0].detach()
            coef_r = sympy.Float(float(w.real.cpu().item()))
            coef_i = sympy.Float(float(w.imag.cpu().item()))
            out += _threshold_multiply(coef_r, symbolic_inputs[i], rounding_decimals)
            out += sympy.I * _threshold_multiply(coef_i, symbolic_inputs[i], rounding_decimals)
        return sympy.sympify(out)

    def prune_by_threshold(self, threshold: float) -> int:
        with torch.no_grad():
            active = self.mask == 1.0
            to_prune = active & (self.weights.abs() < threshold)
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
        symbolic_inputs: List[sympy.Expr],
        rounding_decimals: int = 2,
    ) -> sympy.Expr:
        sym_x0: List[sympy.Expr] = symbolic_inputs
        sym_h: List[sympy.Expr] = self.symbolic_layers[0].get_symbolic_output(sym_x0, rounding_decimals=rounding_decimals)

        for layer in self.symbolic_layers[1:]:
            sym_h = layer.get_symbolic_output(sym_x0 + sym_h, rounding_decimals=rounding_decimals)

        sym_out = self.assembly_layer.get_symbolic_output(sym_x0 + sym_h, rounding_decimals=rounding_decimals)
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
        """
        Returns:
          - list of active edges per symbolic layer (len = n_symbolic_layers)
          - active edges in assembly layer
        """
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
        """
        For each symbolic layer and for the assembly layer, compute an independent threshold
        that prunes approximately `fraction` of that layer's currently-active edges, but
        never prunes below `min_edges_per_layer`.

        Returns:
          - per-symbolic-layer list: (thr, k_to_prune, active_now)
          - assembly tuple: (thr, k_to_prune, active_now)
        """
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

            # kthvalue is 1-indexed in effect: k must be in [1, n]
            kth = torch.kthvalue(mags, k_to_prune).values
            inf = torch.tensor(float("inf"), device=kth.device, dtype=kth.dtype)
            thr = torch.nextafter(kth, inf)  # ensures "< thr" prunes values <= kth
            thr_val = float(max(float(thr.item()), float(eps)))
            return (thr_val, k_to_prune, active_now)

        per_layer: list[tuple[float | None, int, int]] = []
        for layer in self.symbolic_layers:
            m = (layer.mask > 0.5)
            active_now = int(m.sum().item())
            mags = layer.weights.abs()[m].reshape(-1) if active_now > 0 else torch.empty(0, device=layer.weights.device)
            per_layer.append(_layer_threshold(mags, active_now))

        mA = (self.assembly_layer.mask > 0.5)
        activeA = int(mA.sum().item())
        magsA = self.assembly_layer.weights.abs()[mA].reshape(-1) if activeA > 0 else torch.empty(0, device=self.assembly_layer.weights.device)
        asm_tuple = _layer_threshold(magsA, activeA)

        return per_layer, asm_tuple
    
    def prune_by_threshold(self, threshold: float) -> int:
        total = 0
        for layer in self.symbolic_layers:
            total += layer.prune_by_threshold(threshold)
        total += self.assembly_layer.prune_by_threshold(threshold)
        return total

    @torch.no_grad()
    def cascade_threshold_prunning(self, threshold: float, eps: float = 1e-12) -> int:
        def active_edge_mask(w: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            return (m > 0.5) & (w.abs() >= max(threshold, eps))

        n0 = int(self.cfg.n_input_fields)
        L = len(self.symbolic_layers)

        newly_pruned = 0

        asm_w = self.assembly_layer.weights[:, 0]
        asm_m = self.assembly_layer.mask[:, 0]
        keep_asm = active_edge_mask(asm_w, asm_m)

        drop_asm = (asm_m > 0.5) & (~keep_asm)
        if drop_asm.any():
            newly_pruned += int(drop_asm.sum().item())
            self.assembly_layer.mask[drop_asm, 0] = 0.0
            self.assembly_layer.weights[drop_asm, 0] = torch.complex(
                torch.zeros_like(self.assembly_layer.weights.real[drop_asm, 0]),
                torch.zeros_like(self.assembly_layer.weights.imag[drop_asm, 0]),
            )

        last_ops = self.symbolic_layers[-1].n_ops
        req_h_next = keep_asm[n0:n0 + last_ops].clone()

        for layer_idx in range(L - 1, -1, -1):
            layer = self.symbolic_layers[layer_idx]
            req_out = req_h_next.clone()

            for k in range(layer.n_unary_ops):
                if not bool(req_out[k].item()):
                    col = k
                    active_before = (layer.mask[:, col] > 0.5).sum().item()
                    if active_before > 0:
                        newly_pruned += int(active_before)
                        layer.mask[:, col] = 0.0
                        layer.weights[:, col] = torch.complex(
                            torch.zeros_like(layer.weights.real[:, col]),
                            torch.zeros_like(layer.weights.imag[:, col]),
                        )

            for i in range(layer.n_binary_ops):
                out_idx = layer.n_unary_ops + i
                if not bool(req_out[out_idx].item()):
                    a_col = layer.n_unary_ops + 2 * i
                    b_col = a_col + 1

                    active_a = (layer.mask[:, a_col] > 0.5).sum().item()
                    active_b = (layer.mask[:, b_col] > 0.5).sum().item()

                    if active_a > 0:
                        newly_pruned += int(active_a)
                        layer.mask[:, a_col] = 0.0
                        layer.weights[:, a_col] = torch.complex(
                            torch.zeros_like(layer.weights.real[:, a_col]),
                            torch.zeros_like(layer.weights.imag[:, a_col]),
                        )
                    if active_b > 0:
                        newly_pruned += int(active_b)
                        layer.mask[:, b_col] = 0.0
                        layer.weights[:, b_col] = torch.complex(
                            torch.zeros_like(layer.weights.real[:, b_col]),
                            torch.zeros_like(layer.weights.imag[:, b_col]),
                        )

            req_in_this_layer = torch.zeros(layer.n_input_fields, dtype=torch.bool, device=layer.weights.device)

            for k in range(layer.n_unary_ops):
                if not bool(req_out[k].item()):
                    continue
                col = k
                keep_col = active_edge_mask(layer.weights[:, col], layer.mask[:, col])
                drop_col = (layer.mask[:, col] > 0.5) & (~keep_col)
                if drop_col.any():
                    newly_pruned += int(drop_col.sum().item())
                    layer.mask[drop_col, col] = 0.0
                    layer.weights[drop_col, col] = torch.complex(
                        torch.zeros_like(layer.weights.real[drop_col, col]),
                        torch.zeros_like(layer.weights.imag[drop_col, col]),
                    )
                req_in_this_layer |= keep_col

            for i in range(layer.n_binary_ops):
                out_idx = layer.n_unary_ops + i
                if not bool(req_out[out_idx].item()):
                    continue
                a_col = layer.n_unary_ops + 2 * i
                b_col = a_col + 1

                keep_a = active_edge_mask(layer.weights[:, a_col], layer.mask[:, a_col])
                keep_b = active_edge_mask(layer.weights[:, b_col], layer.mask[:, b_col])

                drop_a = (layer.mask[:, a_col] > 0.5) & (~keep_a)
                drop_b = (layer.mask[:, b_col] > 0.5) & (~keep_b)

                if drop_a.any():
                    newly_pruned += int(drop_a.sum().item())
                    layer.mask[drop_a, a_col] = 0.0
                    layer.weights[drop_a, a_col] = torch.complex(
                        torch.zeros_like(layer.weights.real[drop_a, a_col]),
                        torch.zeros_like(layer.weights.imag[drop_a, a_col]),
                    )
                if drop_b.any():
                    newly_pruned += int(drop_b.sum().item())
                    layer.mask[drop_b, b_col] = 0.0
                    layer.weights[drop_b, b_col] = torch.complex(
                        torch.zeros_like(layer.weights.real[drop_b, b_col]),
                        torch.zeros_like(layer.weights.imag[drop_b, b_col]),
                    )

                req_in_this_layer |= (keep_a | keep_b)

            layer.weights.data = nan_to_num_complex(layer.weights.data, nan=0.0, posinf=0.0, neginf=0.0)

            if layer_idx == 0:
                break
            req_h_next = req_in_this_layer[n0:]

        return newly_pruned

    @torch.no_grad()
    def cascade_cleanup_disconnected_(self) -> int:
        """
        Remove (mask=0, weight=0) any edges/nodes that do not contribute
        to the final output, given the CURRENT masks.

        This is the 'levitating nodes' cleanup after any pruning step.
        It does NOT consider magnitudes or thresholds.
        """
        n0 = int(self.cfg.n_input_fields)
        L = len(self.symbolic_layers)

        newly_pruned = 0

        # -------------------------
        # 1) Determine which assembly inputs are required (active)
        # -------------------------
        asm_m = (self.assembly_layer.mask[:, 0] > 0.5)  # shape (n0 + last_ops,)
        # Required last-layer outputs are those assembly edges that connect to them
        last_ops = self.symbolic_layers[-1].n_ops
        req_h_next = asm_m[n0:n0 + last_ops].clone()     # shape (last_ops,)

        # Note: we do NOT drop any assembly edges here; that is handled by your per-layer pruning.
        # This function only removes upstream dead structure.

        # -------------------------
        # 2) Walk backward through symbolic layers
        # -------------------------
        for layer_idx in range(L - 1, -1, -1):
            layer = self.symbolic_layers[layer_idx]

            # req_out: which ops outputs of this layer are required by downstream
            # shape (layer.n_ops,)
            req_out = req_h_next.clone()

            # If an op output is NOT required, the entire operator output column(s) can be dropped:
            #   - unary op k => column k
            #   - binary op i => columns a_col, b_col
            # This removes "dangling operator outputs".
            # We only prune edges that are currently active.
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

            # -------------------------
            # 3) Determine which INPUTS to this layer are required
            # -------------------------
            # A given input feature j is required if it has any active edge into any required op output.
            req_in_this_layer = torch.zeros(layer.n_input_fields, dtype=torch.bool, device=layer.weights.device)

            # unary outputs: column k
            for k in range(layer.n_unary_ops):
                if not bool(req_out[k].item()):
                    continue
                col = k
                keep_edges = (layer.mask[:, col] > 0.5)  # after pruning above
                req_in_this_layer |= keep_edges

            # binary outputs: columns a_col and b_col
            for i in range(layer.n_binary_ops):
                out_idx = layer.n_unary_ops + i
                if not bool(req_out[out_idx].item()):
                    continue
                a_col = layer.n_unary_ops + 2 * i
                b_col = a_col + 1

                keep_a = (layer.mask[:, a_col] > 0.5)
                keep_b = (layer.mask[:, b_col] > 0.5)

                req_in_this_layer |= (keep_a | keep_b)

            layer.weights.data = nan_to_num_complex(layer.weights.data, nan=0.0, posinf=0.0, neginf=0.0)

            # For the previous layer, required outputs are exactly the "h" part of this layer's inputs.
            # Inputs to layer are [x0 (n0), h_prev (...)]
            if layer_idx == 0:
                break
            req_h_next = req_in_this_layer[n0:]

        return newly_pruned
    
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
