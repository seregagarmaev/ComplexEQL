from __future__ import annotations

from typing import List
import torch
import torch.nn as nn

# ============================================================
# Global safety + damping parameters
# ============================================================

# Global clamp for all operator outputs
CLAMP_VAL = 1e15

# Damping: damp(|Im(x)|) = exp(-DAMP_GAMMA * (|Im(x)| + eps)^DAMP_P)
DAMP_GAMMA = 10.0#10.0    10.0 for sin/cos
DAMP_P = 1.0#3.0 # 1.0 for sin/cos
_DAMP_EPS = 1e-12


def nan_to_num_complex(
    x: torch.Tensor,
    nan: float = 0.0,
    posinf: float = CLAMP_VAL,
    neginf: float = -CLAMP_VAL,
) -> torch.Tensor:
    if torch.is_complex(x):
        return torch.complex(
            torch.nan_to_num(x.real, nan=nan, posinf=0.0, neginf=0.0),
            torch.nan_to_num(x.imag, nan=nan, posinf=0.0, neginf=0.0),
        )
    return torch.nan_to_num(x, nan=nan, posinf=0.0, neginf=0.0)


def clamp_complex(
    x: torch.Tensor, min_val: float = -CLAMP_VAL, max_val: float = CLAMP_VAL
) -> torch.Tensor:
    if torch.is_complex(x):
        xr = torch.clamp(x.real, min_val, max_val)
        xi = torch.clamp(x.imag, min_val, max_val)
        return torch.complex(xr, xi)
    return torch.clamp(x, min_val, max_val)


def _sanitize_out(y: torch.Tensor) -> torch.Tensor:
    y = nan_to_num_complex(y, nan=0.0, posinf=0.0, neginf=-0.0)
    y = clamp_complex(y, -CLAMP_VAL, CLAMP_VAL)
    return y


def _real_part(x: torch.Tensor) -> torch.Tensor:
    return x.real if torch.is_complex(x) else x


def _imag_part(x: torch.Tensor) -> torch.Tensor:
    return x.imag if torch.is_complex(x) else torch.zeros_like(x)


def _damp_from_imag(x: torch.Tensor) -> torch.Tensor:
    v = _imag_part(x)
    return torch.exp(-DAMP_GAMMA * torch.pow(v.abs() + _DAMP_EPS, DAMP_P))


# ======================
# UNARY OPERATIONS
# ======================

def identity_operation(x: torch.Tensor) -> torch.Tensor:
    # u = _real_part(x)
    # y = torch.complex(u, torch.zeros_like(u))
    return _sanitize_out(x).unsqueeze(-1)


def const_operation(x: torch.Tensor) -> torch.Tensor:
    u = _real_part(x)
    y = torch.complex(u * 0 + 1, torch.zeros_like(u))
    return _sanitize_out(y).unsqueeze(-1)



def square_operation(x: torch.Tensor) -> torch.Tensor:
    u = _real_part(x)
    damp = _damp_from_imag(x)
    y_real = (u * u) * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)


def cube_operation(x: torch.Tensor) -> torch.Tensor:
    u = _real_part(x)
    damp = _damp_from_imag(x)
    y_real = (u * u * u) * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)


def sqrt_operation(x: torch.Tensor) -> torch.Tensor:
    # y_real = torch.sqrt(x).real
    # y = torch.complex(y_real, torch.zeros_like(y_real))
    y = torch.sqrt(x)
    return _sanitize_out(y).unsqueeze(-1)


def log_operation(x: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    # y_real = torch.log(x).real
    # y = torch.complex(y_real, torch.zeros_like(y_real))
    y = torch.log(x)
    return _sanitize_out(y).unsqueeze(-1)

def exponent_operation(x: torch.Tensor) -> torch.Tensor:
    u = _real_part(x)
    damp = _damp_from_imag(x)
    y_real = torch.exp(u) * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)


def sin_operation(x: torch.Tensor, damp_gamma: float, damp_p: float) -> torch.Tensor:
    u = _real_part(x)
    damp = _damp_from_imag(x)  # uses global DAMP_GAMMA / DAMP_P / _DAMP_EPS
    y_real = torch.sin(u) * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)


def cos_operation(x: torch.Tensor, damp_gamma: float, damp_p: float) -> torch.Tensor:
    u = _real_part(x)
    damp = _damp_from_imag(x)  # uses global DAMP_GAMMA / DAMP_P / _DAMP_EPS
    y_real = torch.cos(u) * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)


def tan_operation(x: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    y_real = torch.tan(x).real
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)





def tanh_operation(x: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    u = _real_part(x)
    damp = _damp_from_imag(x)
    y_real = torch.tanh(u) * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)


# ======================
# BINARY OPERATIONS
# ======================

def multiplication_operation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    y_full = x1.real * x2.real
    damp = _damp_from_imag(x1) * _damp_from_imag(x2)
    y_real = y_full.real * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)


def div_operation(x1: torch.Tensor, x2: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    y_real = (x1 / x2).real
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)


def power_operation(x1: torch.Tensor, x2: torch.Tensor, stair_step_size: float | None = None) -> torch.Tensor:
    y_full = x1 ** x2
    damp = _damp_from_imag(x1) * _damp_from_imag(x2)
    y_real = y_full.real * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return _sanitize_out(y).unsqueeze(-1)


class UnarySurrogate(nn.Module):
    """Wrap an exact unary operation (no learnable params here)."""
    def __init__(
        self,
        operation,
        cfg,
        fname,
        ftype,
        stair_step_size: float | None = None,
        damp_gamma: float | None = None,
        damp_p: float | None = None,
    ):
        super().__init__()
        self.operation = operation
        self.cfg = cfg
        self.fname = fname
        self.ftype = ftype
        self.stair_step_size = stair_step_size
        self.damp_gamma = damp_gamma
        self.damp_p = damp_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stair_step_size is not None:
            out = self.operation(x, self.stair_step_size)
        elif self.damp_gamma is not None:
            out = self.operation(x, self.damp_gamma, self.damp_p)
        else:
            out = self.operation(x)
        return out  # (B,1)


class BinarySurrogate(nn.Module):
    """Wrap an exact binary operation."""
    def __init__(self, operation, cfg, fname, ftype, stair_step_size: float | None = None):
        super().__init__()
        self.operation = operation
        self.cfg = cfg
        self.fname = fname
        self.ftype = ftype
        self.stair_step_size = stair_step_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.stair_step_size is not None:
            out = self.operation(X[:, 0], X[:, 1], self.stair_step_size)
        else:
            out = self.operation(X[:, 0], X[:, 1])
        return out  # (B,1)


def load_models(cfg, layer_idx: int):
    unary_nos = []
    binary_nos = []

    params_list = cfg.no_params_list[layer_idx]

    for params in params_list:
        op = params["library_function"]
        optype = params["library_function_type"]
        stair_step_size = params.get("stair_step_size", None)

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
                model = UnarySurrogate(log_operation, cfg, op, optype, stair_step_size=stair_step_size)
            elif op == "exp":
                model = UnarySurrogate(exponent_operation, cfg, op, optype)
            elif op == "sin":
                model = UnarySurrogate(
                    sin_operation, cfg, op, optype,
                    damp_gamma=params["damp_gamma"], damp_p=params["damp_p"]
                )
            elif op == "cos":
                model = UnarySurrogate(
                    cos_operation, cfg, op, optype,
                    damp_gamma=params["damp_gamma"], damp_p=params["damp_p"]
                )
            elif op == "tan":
                model = UnarySurrogate(tan_operation, cfg, op, optype, stair_step_size=stair_step_size)
            elif op == "tanh":
                model = UnarySurrogate(tanh_operation, cfg, op, optype, stair_step_size=stair_step_size)
            else:
                raise ValueError(f"Unknown exact unary operation '{op}'")
            unary_nos.append(model)

        elif optype == "binary":
            if op == "mul":
                model = BinarySurrogate(multiplication_operation, cfg, op, optype)
            elif op == "div":
                model = BinarySurrogate(div_operation, cfg, op, optype, stair_step_size=stair_step_size)
            elif op == "pow":
                model = BinarySurrogate(power_operation, cfg, op, optype, stair_step_size=stair_step_size)
            else:
                raise ValueError(f"Unknown exact binary operation '{op}'")
            binary_nos.append(model)

        else:
            raise ValueError(f"Unknown library_function_type '{optype}'")

    return unary_nos, binary_nos
