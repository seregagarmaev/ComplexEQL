from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

# Global clamp for all operator outputs
CLAMP_VAL = 1e15


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
    y = nan_to_num_complex(y, nan=0.0, posinf=CLAMP_VAL, neginf=-CLAMP_VAL)
    y = clamp_complex(y, -CLAMP_VAL, CLAMP_VAL)
    return y


# ======================
# UNARY OPERATIONS
# ======================

def identity_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(x)
    return y.unsqueeze(-1)


def const_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(x * 0 + 1)
    return y.unsqueeze(-1)


def square_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(x * x)
    return y.unsqueeze(-1)


def cube_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(x * x * x)
    return y.unsqueeze(-1)


def sqrt_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(torch.sqrt(x))
    return y.unsqueeze(-1)


def log_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(torch.log(x))
    return y.unsqueeze(-1)


def exponent_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(torch.exp(x))
    return y.unsqueeze(-1)


# real version (as you currently use): consumes only real part, returns complex with zero imag
def sin_operation_surrogate(x: torch.Tensor, *, r: float, **_) -> torch.Tensor:
    u = x.real
    r = float(max(min(r, 1.0), 1e-12))

    log_r = torch.log(torch.tensor(r, device=u.device, dtype=u.dtype))
    phi = u.abs() # * u
    damp = torch.exp(log_r * phi)
    y_real = damp * torch.sin(u)

    y = torch.complex(y_real, torch.zeros_like(y_real))
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def cos_operation_surrogate(x: torch.Tensor, *, r: float, **_) -> torch.Tensor:
    u = x.real
    r = float(max(min(r, 1.0), 1e-12))

    log_r = torch.log(torch.tensor(r, device=u.device, dtype=u.dtype))
    phi = u.abs() # * u
    damp = torch.exp(log_r * phi)
    y_real = damp * torch.cos(u)

    y = torch.complex(y_real, torch.zeros_like(y_real))
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def tan_operation(x: torch.Tensor, **_) -> torch.Tensor:
    # simple complex tan (no stair/damping)
    y = _sanitize_out(torch.tan(x))
    return y.unsqueeze(-1)


def tanh_operation(x: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(torch.tanh(x))
    return y.unsqueeze(-1)


# ======================
# BINARY OPERATIONS
# ======================

def multiplication_operation(x1: torch.Tensor, x2: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(x1 * x2)
    return y.unsqueeze(-1)


def div_operation(x1: torch.Tensor, x2: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(x1 / x2)
    return y.unsqueeze(-1)


def power_operation(x1: torch.Tensor, x2: torch.Tensor, **_) -> torch.Tensor:
    y = _sanitize_out(x1 ** x2)
    return y.unsqueeze(-1)


# ======================
# OP WRAPPERS
# ======================

OpParams = Dict[str, Dict[str, Any]]  # e.g. {"sin": {"r": 0.8}, "cos": {"r": 0.9}}


class UnarySurrogate(nn.Module):
    """
    Wrap an exact unary operation (no learnable params).
    Routes op-specific runtime kwargs via `op_params[fname]`.
    """
    def __init__(self, operation, cfg, fname: str):
        super().__init__()
        self.operation = operation
        self.cfg = cfg
        self.fname = fname
        self.ftype = "unary"

    def forward(self, x: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
        kwargs = (op_params or {}).get(self.fname, {})
        return self.operation(x, **kwargs)  # operation decides what it needs


class BinarySurrogate(nn.Module):
    """
    Wrap an exact binary operation (no learnable params).
    Routes op-specific runtime kwargs via `op_params[fname]`.
    """
    def __init__(self, operation, cfg, fname: str):
        super().__init__()
        self.operation = operation
        self.cfg = cfg
        self.fname = fname
        self.ftype = "binary"

    def forward(self, X: torch.Tensor, *, op_params: Optional[OpParams] = None) -> torch.Tensor:
        kwargs = (op_params or {}).get(self.fname, {})
        return self.operation(X[:, 0], X[:, 1], **kwargs)  # (B,1)


# ======================
# MODEL LOADING
# ======================

UNARY_OPS = {
    "id": identity_operation,
    "const": const_operation,
    "square": square_operation,
    "cube": cube_operation,
    "sqrt": sqrt_operation,
    "log": log_operation,
    "exp": exponent_operation,
    "sin": sin_operation_surrogate,
    "cos": cos_operation_surrogate,
    "tan": tan_operation,
    "tanh": tanh_operation,
}

BINARY_OPS = {
    "mul": multiplication_operation,
    "div": div_operation,
    "pow": power_operation,  # include if you later add {"op":"pow","type":"binary"}
}


def load_models(cfg, layer_idx: int):
    """
    Expects cfg.no_params_list[layer_idx] like:
      [{"op":"sin","type":"unary"}, {"op":"div","type":"binary"}, ...]
    """
    unary_nos: list[nn.Module] = []
    binary_nos: list[nn.Module] = []

    specs = cfg.no_params_list[layer_idx]
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
