from __future__ import annotations

from typing import List
import numpy as np
import sympy
import torch
import torch.nn as nn

from src.operations import *
from src.sympy_utils import prune_small_coeff_terms


def _threshold_multiply(coef: sympy.Number, val: sympy.Expr, decimals: int) -> sympy.Expr:
    """
    Multiply `coef * val` but snap to 0 if the rounded coefficient is 0.
    Keeps expressions tidy during symbolic readout.
    """
    try:
        if coef.round(decimals) == 0.0:
            return sympy.Integer(0)
        return (coef * val).n(decimals)
    except Exception:
        return coef * val

def nan_to_num_complex(x: torch.Tensor, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> torch.Tensor:
    """
    torch.nan_to_num for complex tensors on backends where complex is not supported (e.g., MPS).
    Applies nan_to_num to real and imag parts separately.
    """
    if torch.is_complex(x):
        return torch.complex(
            torch.nan_to_num(x.real, nan=nan, posinf=posinf, neginf=neginf),
            torch.nan_to_num(x.imag, nan=nan, posinf=posinf, neginf=neginf),
        )
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


class SymbolicLayer(nn.Module):
    """
    One symbolic layer constituting ComplexEQL:
      - calculates weighted sums of the input variables/functions (complex weights)
      - applies a library of symbolic operations
      - returns per-operation outputs

    Shapes:
      X:   (B, F_in)       (typically real)
      W:   (F_in, F_ops_in), complex
      out: (B, F_ops)      where F_ops = n_unary + n_binary
                           and F_ops_in = n_unary + 2 * n_binary
    """

    def __init__(self, cfg, layer_number: int, n_input_fields: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_number = layer_number

        self.n_input_fields = n_input_fields

        unary_nos, binary_nos = load_models(cfg, layer_number)
        self.unary_nos = nn.ModuleList(unary_nos)
        self.binary_nos = nn.ModuleList(binary_nos)
        self.n_unary_nos = len(self.unary_nos)
        self.n_binary_nos = len(self.binary_nos)
        self.n_ops = self.n_unary_nos + self.n_binary_nos
        self.n_inputs = self.n_unary_nos + 2 * self.n_binary_nos

        # complex weights initialization
        real = (torch.rand(self.n_input_fields, self.n_inputs) - 0.5) * 0.1
        imag = (torch.rand(self.n_input_fields, self.n_inputs) - 0.5) * 0.1
        init_w = torch.complex(real, imag)  # (F_in, n_inputs), complex
        self.weights = nn.Parameter(init_w)
        self.mask = nn.Parameter(torch.ones_like(real), requires_grad=False)  # real mask

        # Freeze operator networks (we only train mixing weights)
        self._freeze_nos()

        # Operator metadata for symbolic readout
        self.function_names = [no.fname for no in self.unary_nos] + [no.fname for no in self.binary_nos]
        self.function_types = [no.ftype for no in self.unary_nos] + [no.ftype for no in self.binary_nos]
        self.functions_dict = cfg.functions_dict

    def _freeze_nos(self) -> None:
        for model in self.unary_nos + self.binary_nos:
            for p in model.parameters():
                p.requires_grad = False

    # # # real only:
    # def get_symbolic_output(
    #     self,
    #     symbolic_inputs: List[sympy.Expr],
    #     rounding_decimals: int = 2,
    # ) -> List[sympy.Expr]:
    #     """
    #     Symbolic mirror of forward() using ONLY REAL PART of complex weights.
    #     Returns list of sympy expressions, one per operator in this layer
    #     (unary first, then binary).
    #     """
    #     outs: List[sympy.Expr] = []

    #     # ---------- UNARY OPS ----------
    #     for op_idx in range(self.n_unary_nos):
    #         op_name = self.function_names[op_idx]
    #         mixed = sympy.Integer(0)

    #         for j in range(len(symbolic_inputs)):
    #             w = self.weights[j, op_idx].detach()
    #             coef_real = sympy.Float(float(w.real.cpu().item()))
    #             mixed += _threshold_multiply(coef_real, symbolic_inputs[j], rounding_decimals)

    #         if op_name == "dydx":
    #             op = self.functions_dict[op_name]
    #             symbolic_out = op(mixed, self.functions_dict["x"])
    #         else:
    #             op = self.functions_dict[op_name]
    #             symbolic_out = op(mixed)

    #         if isinstance(symbolic_out, int) or getattr(symbolic_out, "is_infinite", False):
    #             outs.append(sympy.Integer(0))
    #         else:
    #             outs.append(sympy.sympify(symbolic_out))

    #     # ---------- BINARY OPS ----------
    #     for i in range(self.n_binary_nos):
    #         op_name = self.function_names[self.n_unary_nos + i]
    #         op = self.functions_dict[op_name]

    #         a = sympy.Integer(0)
    #         b = sympy.Integer(0)
    #         a_col = self.n_unary_nos + 2 * i
    #         b_col = a_col + 1

    #         for j in range(len(symbolic_inputs)):
    #             w_a = self.weights[j, a_col].detach()
    #             w_b = self.weights[j, b_col].detach()

    #             coef_a_real = sympy.Float(float(w_a.real.cpu().item()))
    #             coef_b_real = sympy.Float(float(w_b.real.cpu().item()))

    #             a += _threshold_multiply(coef_a_real, symbolic_inputs[j], rounding_decimals)
    #             b += _threshold_multiply(coef_b_real, symbolic_inputs[j], rounding_decimals)

    #         if op_name == "div":
    #             try:
    #                 outs.append(sympy.cancel(a / b))
    #             except Exception:
    #                 outs.append(sympy.sympify(op(a, b)))
    #         else:
    #             outs.append(sympy.sympify(op(a, b)))

    #     return outs

    # real + imag weights:
    def get_symbolic_output(
        self,
        symbolic_inputs: List[sympy.Expr],
        rounding_decimals: int = 2,
    ) -> List[sympy.Expr]:
        """
        Symbolic mirror of forward():
          - mix inputs with (rounded) REAL and IMAG parts of complex weights
          - apply each operator symbolically via functions_dict
        Returns list of sympy expressions, one per operator in this layer (unary first, then binary).
        """
        outs: List[sympy.Expr] = []

        # ---------- UNARY OPS ----------
        for op_idx in range(self.n_unary_nos):
            op_name = self.function_names[op_idx]
            mixed = sympy.Integer(0)

            for j in range(len(symbolic_inputs)):
                w = self.weights[j, op_idx].detach()
                # real part
                coef_real = sympy.Float(float(w.real.cpu().item()))
                mixed += _threshold_multiply(
                    coef_real, symbolic_inputs[j], rounding_decimals
                )
                # imaginary part
                coef_imag = sympy.Float(float(w.imag.cpu().item()))
                mixed += sympy.I * _threshold_multiply(
                    coef_imag, symbolic_inputs[j], rounding_decimals
                )

            if op_name == "dydx":
                op = self.functions_dict[op_name]
                symbolic_out = op(mixed, self.functions_dict["x"])
            else:
                op = self.functions_dict[op_name]
                symbolic_out = op(mixed)

            if isinstance(symbolic_out, int) or getattr(symbolic_out, "is_infinite", False):
                outs.append(sympy.Integer(0))
            else:
                outs.append(sympy.sympify(symbolic_out))

        # ---------- BINARY OPS ----------
        for i in range(self.n_binary_nos):
            op_name = self.function_names[self.n_unary_nos + i]
            op = self.functions_dict[op_name]

            a = sympy.Integer(0)
            b = sympy.Integer(0)
            a_col = self.n_unary_nos + 2 * i
            b_col = a_col + 1

            for j in range(len(symbolic_inputs)):
                w_a = self.weights[j, a_col].detach()
                w_b = self.weights[j, b_col].detach()

                # a: real + i imag
                coef_a_real = sympy.Float(float(w_a.real.cpu().item()))
                coef_a_imag = sympy.Float(float(w_a.imag.cpu().item()))
                a += _threshold_multiply(
                    coef_a_real, symbolic_inputs[j], rounding_decimals
                )
                a += sympy.I * _threshold_multiply(
                    coef_a_imag, symbolic_inputs[j], rounding_decimals
                )

                # b: real + i imag
                coef_b_real = sympy.Float(float(w_b.real.cpu().item()))
                coef_b_imag = sympy.Float(float(w_b.imag.cpu().item()))
                b += _threshold_multiply(
                    coef_b_real, symbolic_inputs[j], rounding_decimals
                )
                b += sympy.I * _threshold_multiply(
                    coef_b_imag, symbolic_inputs[j], rounding_decimals
                )

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
        """
        For each binary op that is 'div', find the maximum |w| among the
        active mixing weights used by numerator and denominator, and divide
        BOTH numerator and denominator mixing weights by this value.

        This keeps (a / b) invariant (since a and b are both scaled),
        but prevents non-identifiability from drifting weights toward zero.

        Returns
        -------
        int
            Number of div-operators that were normalized (i.e., had scale > eps).
        """
        normalized = 0

        if self.n_binary_nos == 0:
            return 0
        
        for i in range(self.n_binary_nos):
            op_name = self.function_names[self.n_unary_nos + i]
            if op_name != "div":
                continue

            a_col = self.n_unary_nos + 2 * i
            b_col = a_col + 1

            m_a = self.mask[:, a_col].bool()
            m_b = self.mask[:, b_col].bool()

            if (not m_a.any()) and (not m_b.any()):
                continue

            w_a = self.weights[:, a_col]
            w_b = self.weights[:, b_col]

            # max magnitude among ACTIVE weights across numerator+denominator
            # max_a = w_a.real[m_a].abs().max() if m_a.any() else torch.tensor(0.0, device=w_a.device)
            # max_b = w_b.real[m_b].abs().max() if m_b.any() else torch.tensor(0.0, device=w_b.device)
            max_a = w_a[m_a].abs().max() if m_a.any() else torch.tensor(0.0, device=w_a.device)
            max_b = w_b[m_b].abs().max() if m_b.any() else torch.tensor(0.0, device=w_b.device)
            scale = torch.maximum(max_a, max_b)

            if scale <= eps:
                continue

            w_a_new = torch.where(m_a, w_a / scale, w_a)
            w_b_new = torch.where(m_b, w_b / scale, w_b)

            self.weights[:, a_col] = w_a_new
            self.weights[:, b_col] = w_b_new

            self.weights.data = nan_to_num_complex(self.weights.data, nan=0.0, posinf=0.0, neginf=0.0)
            normalized += 1

        return normalized

    def weights_for_reg(self) -> torch.Tensor:
        # return self.weights.reshape(-1)
        effective = self.weights * self.mask.to(self.weights.dtype)
        return effective.reshape(-1)

    def lift(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, F_in)
        weights: (F_in, n_inputs), complex
        → (B, n_inputs), complex
        """
        effective_weights = self.weights * self.mask.to(self.weights.dtype)
        return torch.matmul(X.to(effective_weights.dtype), effective_weights)

    def prune_by_threshold(self, threshold: float) -> int:
        """
        Permanently zero weights with |w| < threshold (magnitude).
        Returns number of newly-pruned weights.
        """
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

    def apply_NOs(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply unary operators channelwise and binary operators on pairs of channels.

        X: (B, F_ops_in), complex
        Returns concatenated result: (B, n_unary + n_binary)
        """
        results: List[torch.Tensor] = []

        # unary
        for i, unary_no in enumerate(self.unary_nos):
            out = unary_no(X[:, i])  # (B,) -> (B,1)
            results.append(out)

        # binary
        for i in range(self.n_binary_nos):
            start = self.n_unary_nos + 2 * i
            num = X[:, start]
            den = X[:, start + 1]
            pair = torch.stack((num, den), dim=1)
            out = self.binary_nos[i](pair)
            results.append(out)

        return torch.cat(results, dim=-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, F_in)  ->  (B, n_ops)
        """
        lifted = self.lift(X)        # (B, n_inputs), complex
        out = self.apply_NOs(lifted) # (B, n_ops)
        return out


class AssemblyLayer(nn.Module):
    """
    Final linear combination of all operator outputs from the last symbolic layer:
      Input:  (B, F_last)
      Weights:(F_last, 1), complex
      Output: (B, 1)
    """

    def __init__(self, cfg, last_layer_nos: int) -> None:
        super().__init__()
        self.cfg = cfg

        if getattr(cfg, "weights", None) is not None:
            w_real = torch.tensor(cfg.weights[-1]).float()
            w_imag = torch.zeros_like(w_real)
            init_w = torch.complex(w_real, w_imag)
            self.weights = nn.Parameter(init_w)
            self.mask = nn.Parameter(torch.ones_like(w_real), requires_grad=False)
        else:
            real = (torch.rand(last_layer_nos, 1) - 0.5) * 0.1
            imag = (torch.rand(last_layer_nos, 1) - 0.5) * 0.1
            init_w = torch.complex(real, imag)
            self.weights = nn.Parameter(init_w)
            self.mask = nn.Parameter(torch.ones_like(real), requires_grad=False)

        self.functions_dict = cfg.functions_dict

    # # real only:
    # def get_symbolic_output(
    #     self,
    #     symbolic_inputs: List[sympy.Expr],
    #     rounding_decimals: int = 2,
    # ) -> sympy.Expr:
    #     """
    #     Sum_i Re(w_i) * symbolic_inputs[i] with thresholded rounding
    #     to keep expressions compact.
    #     """
    #     out: sympy.Expr = sympy.Integer(0)
    #     for i in range(len(symbolic_inputs)):
    #         w = self.weights[i, 0].detach()
    #         coef_real = sympy.Float(float(w.real.cpu().item()))
    #         out += _threshold_multiply(coef_real, symbolic_inputs[i], rounding_decimals)
    #     return sympy.sympify(out)

    # real + imag weights:
    def get_symbolic_output(
        self,
        symbolic_inputs: List[sympy.Expr],
        rounding_decimals: int = 2,
    ) -> sympy.Expr:
        """
        Sum_i (Re(w_i) + i Im(w_i)) * symbolic_inputs[i] with thresholded rounding
        to keep expressions compact.
        """
        out: sympy.Expr = sympy.Integer(0)
        for i in range(len(symbolic_inputs)):
            w = self.weights[i, 0].detach()

            # real part
            coef_real = sympy.Float(float(w.real.cpu().item()))
            out += _threshold_multiply(
                coef_real, symbolic_inputs[i], rounding_decimals
            )

            # imaginary part
            coef_imag = sympy.Float(float(w.imag.cpu().item()))
            out += sympy.I * _threshold_multiply(
                coef_imag, symbolic_inputs[i], rounding_decimals
            )

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
        """
        Flattened EFFECTIVE weights for regularization (includes pruning mask).
        Use .abs() outside for magnitude-based penalties.
        """
        effective = self.weights * self.mask.to(self.weights.dtype)
        return effective.reshape(-1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, F_last)
        weights: (F_last, 1), complex
        → (B, 1), complex
        """
        effective_weights = self.weights * self.mask.to(self.weights.dtype)
        result = torch.matmul(X, effective_weights)  # (B, 1)
        return result


class ComplexEQL(nn.Module):
    """
    ComplexEQL model: stacks SymbolicLayer blocks and a final AssemblyLayer.

    Skip-connection mapping:
      - layer 0 input: raw inputs x0
      - layer l>=1 input: concat([x0, h_{l-1}])
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        layers: list[SymbolicLayer] = []
        prev_n_ops: int | None = None

        for layer_idx in range(cfg.n_symbolic_layers):
            if layer_idx == 0:
                n_in = int(cfg.n_input_fields)
            else:
                n_in = int(cfg.n_input_fields) + int(prev_n_ops)
            
            layer = SymbolicLayer(cfg, layer_idx, n_input_fields=n_in).to(cfg.device)
            layers.append(layer)
            prev_n_ops = layer.n_ops

        self.symbolic_layers = nn.ModuleList(layers)
        last_outputs = self.symbolic_layers[-1].n_ops
        assembly_in_dim = int(cfg.n_input_fields) + int(last_outputs)
        self.assembly_layer = AssemblyLayer(cfg, assembly_in_dim)

    def get_symbolic_expression(
        self,
        symbolic_inputs: List[sympy.Expr],
        rounding_decimals: int = 2,
    ) -> sympy.Expr:
        """
        Propagate SymPy input variables through all layers (using Re(weights))
        to get a final real-valued symbolic expression, mirroring skip-connections.
        """
        sym_x0: List[sympy.Expr] = symbolic_inputs
        sym_h: List[sympy.Expr] = self.symbolic_layers[0].get_symbolic_output(sym_x0, rounding_decimals=rounding_decimals)

        for layer in self.symbolic_layers[1:]:
            sym_layer_in = sym_x0 + sym_h
            sym_h = layer.get_symbolic_output(sym_layer_in, rounding_decimals=rounding_decimals)
        
        sym_asm_in = sym_x0 + sym_h
        sym_out = self.assembly_layer.get_symbolic_output(sym_asm_in, rounding_decimals=rounding_decimals)
        return prune_small_coeff_terms(sym_out, rounding_decimals)

    def get_real_weights_list(self) -> list[torch.Tensor]:
        """
        Return a flat list of 1D REAL tensors: real parts of all complex weights
        (symbolic layers + assembly layer), for use in real-valued regularization.
        """
        weights_list: list[torch.Tensor] = []

        for layer in self.symbolic_layers:
            effective = layer.weights * layer.mask.to(layer.weights.dtype)
            weights_list.append(effective.real.view(-1))

        effective = self.assembly_layer.weights * self.assembly_layer.mask.to(self.assembly_layer.weights.dtype)
        weights_list.append(effective.real.view(-1))

        return weights_list

    def get_imag_weights_list(self) -> list[torch.Tensor]:
        """
        Return a flat list of 1D REAL tensors: imaginary parts of all complex weights
        (symbolic layers + assembly layer), for use in imaginary-part regularization.
    
        Each tensor is detached view of the imaginary components.
        """
        weights_list: list[torch.Tensor] = []

        for layer in self.symbolic_layers:
            effective = layer.weights * layer.mask.to(layer.weights.dtype)
            weights_list.append(effective.imag.view(-1))

        effective = self.assembly_layer.weights * self.assembly_layer.mask.to(self.assembly_layer.weights.dtype)
        weights_list.append(effective.imag.view(-1))

        return weights_list

    @torch.no_grad()
    def count_active_edges(self) -> int:
        """
        Number of currently non-pruned edges across the entire graph
        (symbolic mixing edges + assembly edges), as defined by masks.
        """
        total = 0
        for layer in self.symbolic_layers:
            total += int((layer.mask > 0.5).sum().item())
        total += int((self.assembly_layer.mask > 0.5).sum().item())
        return total
    
    @torch.no_grad()
    def pruning_threshold_from_fraction(
        self,
        fraction: float,
        *,
        min_edges_total: int,
        eps: float = 1e-12,
    ) -> tuple[float | None, int, int]:
        """
        Compute a magnitude threshold so that approximately `fraction` of the
        currently active edges are pruned (smallest magnitudes), but never
        reduce the total active edges below `min_edges_total`.

        Returns:
            (threshold | None, k_to_prune, active_edges)

        Notes:
        - We compute the k-th smallest magnitude among ACTIVE edges.
        - We use nextafter(kth, +inf) so that edges with |w| == kth are also pruned
          by your strict '< threshold' logic inside cascade_threshold_prunning().
        """
        fraction = float(fraction)
        if fraction <= 0.0:
            active = self.count_active_edges()
            return (None, 0, active)

        active = self.count_active_edges()
        if active <= int(min_edges_total):
            return (None, 0, active)

        # collect magnitudes of ACTIVE edges only
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

        # target prune count, but capped so we never go below min_edges_total
        k_target = int(np.floor(fraction * n))
        k_cap = n - int(min_edges_total)
        k_to_prune = max(0, min(k_target, k_cap))

        if k_to_prune <= 0:
            return (None, 0, active)

        # kthvalue is 1-indexed
        kth = torch.kthvalue(mags, k_to_prune).values

        # ensure we prune weights with magnitude == kth as well
        inf = torch.tensor(float("inf"), device=kth.device, dtype=kth.dtype)
        thr = torch.nextafter(kth, inf)

        # also respect eps (your cascade uses max(threshold, eps))
        thr_val = float(max(float(thr.item()), float(eps)))
        return (thr_val, k_to_prune, active)
    
    def prune_by_threshold(self, threshold: float) -> int:
        total = 0
        for layer in self.symbolic_layers:
            total += layer.prune_by_threshold(threshold)
        total += self.assembly_layer.prune_by_threshold(threshold)
        return total
    
    @torch.no_grad()
    def cascade_threshold_prunning(self, threshold: float, eps: float = 1e-12) -> int:
        """
        Cascading pruning from the output node:
          - Start from assembly weights: keep only inputs with |w| >= threshold
          - Backpropagate "required" operator outputs layer-by-layer
          - For required outputs, prune their incoming mixing edges with |w| < threshold
          - Any operator output not required is fully pruned (its input columns masked out)

        Returns number of mask entries newly set to 0.
        """

        def active_edge_mask(w: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            return (m > 0.5) & (w.abs() >= max(threshold, eps))

        n0 = int(self.cfg.n_input_fields)
        L = len(self.symbolic_layers)

        newly_pruned = 0

        # ---------- 1) Assembly: decide which inputs are kept ----------
        asm_w = self.assembly_layer.weights[:, 0]
        asm_m = self.assembly_layer.mask[:, 0]
        keep_asm = active_edge_mask(asm_w, asm_m)  # (n0 + last_ops,)

        # prune assembly inputs that are not kept
        drop_asm = (asm_m > 0.5) & (~keep_asm)
        if drop_asm.any():
            newly_pruned += int(drop_asm.sum().item())
            self.assembly_layer.mask[drop_asm, 0] = 0.0
            self.assembly_layer.weights[drop_asm, 0] = torch.complex(
                torch.zeros_like(self.assembly_layer.weights.real[drop_asm, 0]),
                torch.zeros_like(self.assembly_layer.weights.imag[drop_asm, 0]),
            )

        last_ops = self.symbolic_layers[-1].n_ops
        req_h_next = keep_asm[n0:n0 + last_ops].clone()  # required outputs of last symbolic layer

        # ---------- 2) Backward cascade through symbolic layers ----------
        for layer_idx in range(L - 1, -1, -1):
            layer = self.symbolic_layers[layer_idx]
            req_out = req_h_next.clone()  # (layer.n_ops,)

            # 2a) prune operators not required (kill whole operator inputs)
            # unary k -> column k
            for k in range(layer.n_unary_nos):
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

            # binary i -> columns (a_col, b_col)
            for i in range(layer.n_binary_nos):
                out_idx = layer.n_unary_nos + i
                if not bool(req_out[out_idx].item()):
                    a_col = layer.n_unary_nos + 2 * i
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

            # 2b) for required operators, prune their incoming mixing edges below threshold
            # and compute which inputs to this layer are needed by the kept edges
            req_in_this_layer = torch.zeros(layer.n_input_fields, dtype=torch.bool, device=layer.weights.device)

            # unary
            for k in range(layer.n_unary_nos):
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

            # binary
            for i in range(layer.n_binary_nos):
                out_idx = layer.n_unary_nos + i
                if not bool(req_out[out_idx].item()):
                    continue
                a_col = layer.n_unary_nos + 2 * i
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

            # propagate required outputs to previous layer (skip connection layout)
            if layer_idx == 0:
                break
            req_h_next = req_in_this_layer[n0:]  # required outputs of previous layer

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
        """
        Multiply imaginary parts of ALL effective weights by `coeff`.
        Real parts unchanged. Respects masks (masked weights remain zero anyway).
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F0) -> y: (B, 1), complex

        Skip-connection mapping:
          layer 0: h = layer0(x0)
          layer l>=1: h = layerl(concat([x0, h]))
        """
        x0 = x
        h = self.symbolic_layers[0](x0)

        x0_c = x0.to(h.dtype)
        for layer in self.symbolic_layers[1:]:
            layer_in = torch.cat([x0_c, h], dim=1)
            h = layer(layer_in)

        assembly_in = torch.cat([x0_c, h], dim=1)
        y = self.assembly_layer(assembly_in)
        return y
