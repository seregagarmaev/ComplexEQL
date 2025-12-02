from __future__ import annotations

from typing import List
import torch
import torch.nn as nn


def identity_operation(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1)  # (B,) -> (B,1)


def const_operation(x: torch.Tensor) -> torch.Tensor:
    return (x * 0 + 1).unsqueeze(-1)  # (B,) -> (B,1)


def square_operation(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = x * x
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def cube_operation(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = x * x * x
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def sqrt_operation(x: torch.Tensor) -> torch.Tensor:
    # true complex sqrt
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.sqrt(x)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def log_operation(x: torch.Tensor) -> torch.Tensor:
    # complex logarithm
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.log(x)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def exponent_operation(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.exp(x)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def sin_operation(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.sin(x)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def cos_operation(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.cos(x)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def multiplication_operation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
    x2 = torch.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
    out = x1 * x2
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def div_operation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # plain complex division
    x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
    x2 = torch.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
    out = x1 / x2
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def power_operation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    x1^x2 as a binary operation on complex tensors.
    """
    x1 = torch.nan_to_num(x1, nan=0.0, posinf=0.0, neginf=0.0)
    x2 = torch.nan_to_num(x2, nan=0.0, posinf=0.0, neginf=0.0)
    out = x1 ** x2
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


class UnarySurrogate(nn.Module):
    """Wrap an exact unary operation (no learnable params)."""
    def __init__(self, operation, cfg, fname, ftype):
        super().__init__()
        self.operation = operation
        self.cfg = cfg
        self.fname = fname
        self.ftype = ftype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,)
        out = self.operation(x)
        return out  # (B,1)


class BinarySurrogate(nn.Module):
    """Wrap an exact binary operation."""
    def __init__(self, operation, cfg, fname, ftype):
        super().__init__()
        self.operation = operation
        self.cfg = cfg
        self.fname = fname
        self.ftype = ftype

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, 2)
        out = self.operation(X[:, 0], X[:, 1])
        return out  # (B,1)


def load_models(cfg, layer_idx: int):
    """
    Build the list of exact symbolic operations for a given symbolic layer.

    Supported 'model_type':
      - 'exact'  : uses exact versions of operations (complex-capable)
    """
    unary_nos = []
    binary_nos = []

    params_list = cfg.no_params_list[layer_idx]

    for params in params_list:
        model_type = params["model_type"]
        op = params["library_function"]
        optype = params["library_function_type"]

        if model_type != "exact":
            raise ValueError(
                f"ComplexEQL now only supports exact operations. "
                f"Got model_type='{model_type}' for op '{op}'."
            )

        if optype == "unary":
            if op == "id":
                model = UnarySurrogate(identity_operation, cfg, op, optype)
            elif op == "const":
                model = UnarySurrogate(const_operation, cfg, op, optype)
            elif op == "square":
                model = UnarySurrogate(square_operation, cfg, op, optype)
            elif op == "cube":
                model = UnarySurrogate(cube_operation, cfg, op, optype)
            elif op == "sqrt":
                model = UnarySurrogate(sqrt_operation, cfg, op, optype)
            elif op == "log":
                model = UnarySurrogate(log_operation, cfg, op, optype)
            elif op == "exp":
                model = UnarySurrogate(exponent_operation, cfg, op, optype)
            elif op == "sin":
                model = UnarySurrogate(sin_operation, cfg, op, optype)
            elif op == "cos":
                model = UnarySurrogate(cos_operation, cfg, op, optype)
            else:
                raise ValueError(f"Unknown exact unary operation '{op}'")
            unary_nos.append(model)

        elif optype == "binary":
            if op == "mul":
                model = BinarySurrogate(multiplication_operation, cfg, op, optype)
            elif op == "div":
                model = BinarySurrogate(div_operation, cfg, op, optype)
            elif op in ("pow", "abs_pow"):  # treat old 'abs_pow' name as power now
                model = BinarySurrogate(power_operation, cfg, op, optype)
            else:
                raise ValueError(f"Unknown exact binary operation '{op}'")
            binary_nos.append(model)
        else:
            raise ValueError(f"Unknown library_function_type '{optype}'")

    return unary_nos, binary_nos