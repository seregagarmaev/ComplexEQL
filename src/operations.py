from __future__ import annotations

import math
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

# Global clamp for all operator outputs
CLAMP_VAL = 1e10


def nan_to_num_complex(
    x: torch.Tensor,
    nan: float = 0.0,
    posinf: float = CLAMP_VAL,
    neginf: float = -CLAMP_VAL,
) -> torch.Tensor:
    if torch.is_complex(x):
        return torch.complex(
            torch.nan_to_num(x.real, nan=nan, posinf=posinf, neginf=neginf),
            torch.nan_to_num(x.imag, nan=nan, posinf=posinf, neginf=neginf),
        )
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def clamp_complex(
    x: torch.Tensor, min_val: float = -CLAMP_VAL, max_val: float = CLAMP_VAL
) -> torch.Tensor:
    if torch.is_complex(x):
        xr = torch.clamp(x.real, min_val, max_val)
        xi = torch.clamp(x.imag, min_val, max_val)
        return torch.complex(xr, xi)
    return torch.clamp(x, min_val, max_val)


def _sanitize_out(y: torch.Tensor) -> torch.Tensor:
    y = nan_to_num_complex(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = clamp_complex(y, -CLAMP_VAL, CLAMP_VAL)
    return y


# ======================
# UNARY OPERATIONS
# ======================

def identity_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y_real = x.real
    y = torch.complex(y_real, y_real.new_zeros(y_real.shape))
    return y.unsqueeze(-1)


def const_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y = x.real.new_ones(x.real.shape)
    y = torch.complex(y, y.new_zeros(y.shape))
    return y.unsqueeze(-1)


def square_operation(x: torch.Tensor, **_) -> torch.Tensor:
    u = x.real
    y_real = u * u
    y = torch.complex(y_real, y_real.new_zeros(y_real.shape))
    return y.unsqueeze(-1)


def log_operation(x: torch.Tensor, **_) -> torch.Tensor:
    z = torch.log(x + 1.0)
    # z = torch.log(x)
    y_real = z.real #torch.abs(z)
    y = torch.complex(y_real, y_real.new_zeros(y_real.shape))
    return y.unsqueeze(-1)


def log10_operation(x: torch.Tensor, **_) -> torch.Tensor:
    z = torch.log(x + 1.0) / math.log(10.0)
    y_real = z.real
    y = torch.complex(y_real, y_real.new_zeros(y_real.shape))
    return y.unsqueeze(-1)


def sqrt_operation(x: torch.Tensor, **_) -> torch.Tensor:
    z = torch.sqrt(x)
    y_real = z.real # torch.abs(z)
    y = torch.complex(y_real, y_real.new_zeros(y_real.shape))
    return y.unsqueeze(-1)


def exponent_operation(x: torch.Tensor, **_) -> torch.Tensor:
    u = x.real
    y_real = torch.exp(u)
    y = torch.complex(y_real, y_real.new_zeros(y_real.shape))
    return y.unsqueeze(-1)


# ======================
# BINARY OPERATIONS
# ======================

def multiplication_operation(x1: torch.Tensor, x2: torch.Tensor, **_) -> torch.Tensor:
    y_real = x1.real * x2.real
    y = torch.complex(y_real, y_real.new_zeros(y_real.shape))
    return y.unsqueeze(-1)


def div_operation(x1: torch.Tensor, x2: torch.Tensor, **_) -> torch.Tensor:
    q = x1.real / x2
    y_real = q.real
    y = torch.complex(y_real, y_real.new_zeros(y_real.shape))
    return y.unsqueeze(-1)


def pow_operation(x: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
    base = x + 1.0
    log_base = torch.log(base).real
    expo = y.real * log_base
    out_real = torch.exp(expo)
    out = torch.complex(out_real, out_real.new_zeros(out_real.shape))
    return out.unsqueeze(-1)


def resonator_operation(x1: torch.Tensor, x2: torch.Tensor, **_) -> torch.Tensor:
    q = x1.real / (x2 * x2)
    y_real = q.real
    y = torch.complex(y_real, y_real.new_zeros(y_real.shape))
    return y.unsqueeze(-1)

# ======================
# OP WRAPPERS
# ======================

OpParams = Dict[str, Dict[str, Any]]


class UnarySurrogate(nn.Module):
    def __init__(self, operation, cfg, fname: str):
        super().__init__()
        self.operation = operation
        self.cfg = cfg
        self.fname = fname
        self.ftype = "unary"

    def forward(self, x: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
        kwargs = (op_params or {}).get(self.fname, {})
        return self.operation(x, **kwargs)


class BinarySurrogate(nn.Module):
    def __init__(self, operation, cfg, fname: str):
        super().__init__()
        self.operation = operation
        self.cfg = cfg
        self.fname = fname
        self.ftype = "binary"

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
        kwargs = (op_params or {}).get(self.fname, {})
        return self.operation(x1, x2, **kwargs)


# ======================
# MODEL LOADING
# ======================

UNARY_OPS = {
    "id": identity_operation,
    "const": const_operation,
    "square": square_operation,
    "sqrt": sqrt_operation,
    "log": log_operation,
    "log10": log10_operation,
    "exp": exponent_operation,
}

BINARY_OPS = {
    "mul": multiplication_operation,
    "div": div_operation,
    "x^y": pow_operation,
    "resonator": resonator_operation,
}


def load_models(cfg, layer_idx: int, specs_override=None):
    unary_nos: list[nn.Module] = []
    binary_nos: list[nn.Module] = []

    specs = specs_override if specs_override is not None else cfg.no_params_list[layer_idx]
    for spec in specs:
        op = spec["op"]
        typ = spec["type"]

        if typ == "unary":
            if op not in UNARY_OPS:
                raise ValueError(f"Unknown unary op '{op}'")
            unary_nos.append(UnarySurrogate(UNARY_OPS[op], cfg, op))

        elif typ == "binary":
            if op not in BINARY_OPS:
                raise ValueError(f"Unknown binary op '{op}'")
            binary_nos.append(BinarySurrogate(BINARY_OPS[op], cfg, op))

        else:
            raise ValueError(f"Unknown op type '{typ}' (op='{op}')")

    return unary_nos, binary_nos
