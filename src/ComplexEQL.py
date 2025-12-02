from __future__ import annotations

from typing import List
import numpy as np
import sympy
import torch
import torch.nn as nn

from src.operations import *


def _threshold_multiply(coef: sympy.Number, val: sympy.Expr, decimals: int) -> sympy.Expr:
    """
    Multiply `coef * val` but snap to 0 if the rounded coefficient is 0.
    Keeps expressions tidy during symbolic readout.
    """
    try:
        if coef.round(decimals=decimals) == 0:
            return sympy.Integer(0)
        return (coef * val).n(decimals)
    except Exception:
        return coef * val



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

    def __init__(self, cfg, layer_number: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_number = layer_number

        self.n_input_fields = (
            cfg.n_input_fields if layer_number == 0 else len(cfg.no_params_list[layer_number - 1])
        )

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
        self.function_names = [no.fname for no in self.unary_nos] + [
            no.fname for no in self.binary_nos
        ]
        self.function_types = [no.ftype for no in self.unary_nos] + [
            no.ftype for no in self.binary_nos
        ]
        self.functions_dict = cfg.functions_dict

    def _freeze_nos(self) -> None:
        for model in self.unary_nos + self.binary_nos:
            for p in model.parameters():
                p.requires_grad = False

    def get_symbolic_output(
        self,
        symbolic_inputs: List[sympy.Expr],
        rounding_decimals: int = 2,
    ) -> List[sympy.Expr]:
        """
        Symbolic mirror of forward():
          - mix inputs with (rounded) REAL PART of complex weights
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
                coef_real = float(w.real.cpu().item())
                coef_sym = sympy.Float(coef_real)
                mixed += _threshold_multiply(coef_sym, symbolic_inputs[j], rounding_decimals)

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

                coef_a = sympy.Float(float(w_a.real.cpu().item()))
                coef_b = sympy.Float(float(w_b.real.cpu().item()))

                a += _threshold_multiply(coef_a, symbolic_inputs[j], rounding_decimals)
                b += _threshold_multiply(coef_b, symbolic_inputs[j], rounding_decimals)

            if op_name == "div":
                try:
                    outs.append(sympy.cancel(a / b))
                except Exception:
                    outs.append(sympy.sympify(op(a, b)))
            else:
                outs.append(sympy.sympify(op(a, b)))

        return outs

    def weights_for_reg(self) -> torch.Tensor:
        """
        Flattened weights for regularization (magnitude-based, handled outside via .abs()).
        """
        return self.weights.reshape(-1)

    def lift(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, F_in)
        weights: (F_in, n_inputs), complex
        → (B, n_inputs), complex
        """
        effective_weights = self.weights * self.mask.to(self.weights.dtype)
        return torch.matmul(X, effective_weights)

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

            self.weights.data = torch.nan_to_num(
                self.weights.data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

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
            out = unary_no(X[:, i])  # X[:, i]: (B,)
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise RuntimeError(f"NaN/Inf in unary op {i} (layer {self.layer_number})")
            results.append(out)  # (B,1)

        # binary
        for i in range(self.n_binary_nos):
            start = self.n_unary_nos + 2 * i
            num = X[:, start]       # (B,)
            den = X[:, start + 1]   # (B,)

            pair = torch.stack((num, den), dim=1)  # (B, 2)
            out = self.binary_nos[i](pair)         # (B,1)

            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"[DEBUG] layer {self.layer_number}, binary {i}")
                print("  num stats:", num.min().item(), num.max().item())
                print("  den stats:", den.min().item(), den.max().item())
                print("  out stats:", out.min().item(), out.max().item())
                raise RuntimeError(f"NaN/Inf in binary op {i} (layer {self.layer_number})")

            results.append(out)

        return torch.cat(results, dim=-1)  # (B, n_ops)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, F_in)  ->  (B, n_ops)
        """
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise RuntimeError(f"NaN/Inf in inputs at layer {self.layer_number}")

        active = self.mask == 1.0
        if active.any():
            w_active = self.weights[active]
            if torch.isnan(w_active).any() or torch.isinf(w_active).any():
                raise RuntimeError(f"NaN/Inf in ACTIVE weights at layer {self.layer_number}")

        lifted = self.lift(X)        # (B, F_ops_in), complex
        out = self.apply_NOs(lifted) # (B, n_ops)

        if torch.isnan(out).any() or torch.isinf(out).any():
            raise RuntimeError(f"NaN/Inf in outputs at layer {self.layer_number}")

        return out



class AssemblyLayer(nn.Module):
    """
    Final linear combination of all operator outputs from the last symbolic layer:
      Input:  (B, F_last)
      Weights:(F_last, 1), complex
      Output: (B, 1)
    Also provides a symbolic readout that mirrors this linear combination
    using the REAL PART of the complex weights.
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

    def get_symbolic_output(
        self,
        symbolic_inputs: List[sympy.Expr],
        rounding_decimals: int = 2,
    ) -> sympy.Expr:
        """
        Sum_i Re(w_i) * symbolic_inputs[i] with thresholded rounding
        to keep expressions compact.
        """
        out: sympy.Expr = sympy.Integer(0)
        for i in range(len(symbolic_inputs)):
            w = self.weights[i, 0].detach()
            coef_real = float(w.real.cpu().item())
            coef_sym = sympy.Float(coef_real)
            out += _threshold_multiply(coef_sym, symbolic_inputs[i], rounding_decimals)
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

            self.weights.data = torch.nan_to_num(
                self.weights.data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        return num_pruned

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, F_last)
        weights: (F_last, 1), complex
        → (B, 1), complex
        """
        effective_weights = self.weights * self.mask.to(self.weights.dtype)
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise RuntimeError("NaN/Inf in X just before assembly matmul")
        if torch.isnan(effective_weights).any() or torch.isinf(effective_weights).any():
            raise RuntimeError("NaN/Inf in effective_weights in AssemblyLayer")

        result = torch.matmul(X, effective_weights)  # (B, 1)
        return result



class ComplexEQL(nn.Module):
    """
    ComplexEQL model: stacks SymbolicLayer blocks and a final AssemblyLayer.

    Overall mapping:
      x: (B, F_in) -> y: (B, 1)   (y is complex; you can take y.real for real-valued outputs)
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.symbolic_layers = nn.ModuleList(
            [SymbolicLayer(cfg, i).to(cfg.device) for i in range(cfg.n_symbolic_layers)]
        )
        last_outputs = self.symbolic_layers[-1].n_ops
        self.assembly_layer = AssemblyLayer(cfg, last_outputs)

    def get_symbolic_expression(
        self,
        symbolic_inputs: List[sympy.Expr],
        rounding_decimals: int = 2,
    ) -> sympy.Expr:
        """
        Propagate SymPy input variables through all layers (using Re(weights))
        to get a final real-valued symbolic expression.
        """
        sym_out: List[sympy.Expr] | sympy.Expr = symbolic_inputs
        for layer in self.symbolic_layers:
            sym_out = layer.get_symbolic_output(sym_out, rounding_decimals=rounding_decimals)
        return self.assembly_layer.get_symbolic_output(sym_out, rounding_decimals=rounding_decimals)

    def get_weights_list(self) -> list[torch.Tensor]:
        """
        Return a flat list of 1D tensors: complex mixing weights of each layer
        and complex assembly weights.
        """
        weights_list: list[torch.Tensor] = []

        for layer in self.symbolic_layers:
            weights_list.append(layer.weights.view(-1))

        weights_list.append(self.assembly_layer.weights.view(-1))  # complex

        return weights_list

    def prune_by_threshold(self, threshold: float) -> int:
        total = 0
        for layer in self.symbolic_layers:
            total += layer.prune_by_threshold(threshold)
        total += self.assembly_layer.prune_by_threshold(threshold)
        return total

    def sanitize_gradients(self, max_grad: float = 1e3) -> None:
        """
        Replace NaN/Inf in grads and clamp magnitude for complex gradients.
        Call after loss.backward() and before optimizer.step().
        """
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is None:
                    continue
                g = p.grad.data
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                mag = torch.abs(g)
                mask = mag > max_grad
                if mask.any():
                    scale = max_grad / mag[mask]
                    g = g.clone()
                    g[mask] = g[mask] * scale
                p.grad.data.copy_(g)

    def sanitize_weights(self, clamp_value: float = 1e6) -> None:
        """
        Clean NaN/Inf in complex weights and clamp magnitude.
        (This is optimization safety, not part of the symbolic operations.)
        """
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
                w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                w = _clamp_complex_by_magnitude(w, clamp_value)
                layer.weights.data.copy_(w)

            w = self.assembly_layer.weights.data
            w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            w = _clamp_complex_by_magnitude(w, clamp_value)
            self.assembly_layer.weights.data.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F_in) -> y: (B, 1), complex
        """
        for layer in self.symbolic_layers:
            x = layer(x)
        y = self.assembly_layer(x)
        return y
