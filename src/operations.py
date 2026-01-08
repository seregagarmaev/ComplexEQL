from __future__ import annotations

from typing import List
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


def clamp_complex(x: torch.Tensor, min_val: float = -CLAMP_VAL, max_val: float = CLAMP_VAL) -> torch.Tensor:
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

def identity_operation(x: torch.Tensor) -> torch.Tensor:
    y = x
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def const_operation(x: torch.Tensor) -> torch.Tensor:
    y = x * 0 + 1
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def square_operation(x: torch.Tensor) -> torch.Tensor:
    y = x * x
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def cube_operation(x: torch.Tensor) -> torch.Tensor:
    y = x * x * x
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def sqrt_operation(x: torch.Tensor) -> torch.Tensor:
    y = torch.sqrt(x)
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def log_operation(x: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    # stair_step_size kept for interface compatibility; unused
    y = torch.log(x)
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def exponent_operation(x: torch.Tensor) -> torch.Tensor:
    y = torch.exp(x)
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


# def sin_operation(x: torch.Tensor, damp_gamma: float, damp_p: float) -> torch.Tensor:
#     u = x.real
#     v = x.imag
#     eps = 1e-12
#     damp = torch.exp(-damp_gamma * torch.pow(v.abs() + eps, damp_p))
#     y_real = torch.sin(u) * damp
#     y = torch.complex(y_real, torch.zeros_like(y_real))
#     y = _sanitize_out(y)
#     return y.unsqueeze(-1)

# # complex verison
# def sin_operation_surrogate(x: torch.Tensor, r: float) -> torch.Tensor:
#     u = x  # keep complex
#     r = float(max(min(r, 1.0), 1e-12))

#     log_r = torch.log(torch.tensor(r, device=u.device, dtype=u.real.dtype))

#     phi = u.real * u.real + u.imag * u.imag
#     damp = torch.exp(log_r * phi)          # real tensor
#     y = damp.to(u.dtype) * torch.sin(u)    # complex tensor

#     y = _sanitize_out(y)
#     return y.unsqueeze(-1)

# real version
def sin_operation_surrogate(x: torch.Tensor, r: float) -> torch.Tensor:
    u = x.real  # use only real part
    r = float(max(min(r, 1.0), 1e-12))

    log_r = torch.log(torch.tensor(r, device=u.device, dtype=u.dtype))

    phi = u * u
    damp = torch.exp(log_r * phi)          # real tensor
    y_real = damp * torch.sin(u)           # real tensor

    y = torch.complex(y_real, torch.zeros_like(y_real))  # complex with zero imag
    y = _sanitize_out(y)
    return y.unsqueeze(-1)

def cos_operation(x: torch.Tensor, damp_gamma: float, damp_p: float) -> torch.Tensor:
    u = x.real
    v = x.imag
    eps = 1e-12
    damp = torch.exp(-damp_gamma * torch.pow(v.abs() + eps, damp_p))
    y_real = torch.cos(u) * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def tan_operation(x: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    # stair_step_size kept for interface compatibility; unused
    y = torch.tan(x)
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def tanh_operation(x: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    # stair_step_size kept for interface compatibility; unused
    y = torch.tanh(x)
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


# ======================
# BINARY OPERATIONS
# ======================

def multiplication_operation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    y = x1 * x2
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def div_operation(x1: torch.Tensor, x2: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    # stair_step_size kept for interface compatibility; unused
    y = x1 / x2
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


def power_operation(x1: torch.Tensor, x2: torch.Tensor, stair_step_size: float | None = None) -> torch.Tensor:
    # stair_step_size kept for interface compatibility; unused
    y = x1 ** x2
    y = _sanitize_out(y)
    return y.unsqueeze(-1)


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

    # def forward(self, x: torch.Tensor, **runtime_kwargs) -> torch.Tensor:
    # # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     if self.stair_step_size is not None:
    #         out = self.operation(x, self.stair_step_size)
    #     elif self.damp_gamma is not None:
    #         out = self.operation(x, self.damp_gamma, self.damp_p)
    #     else:
    #         # out = self.operation(x)
    #         out = self.operation(x, **runtime_kwargs) if runtime_kwargs else self.operation(x)
    #     return out  # (B,1)
    def forward(self, x: torch.Tensor, **runtime_kwargs) -> torch.Tensor:
        # Priority: explicit params that define the op's fixed signature
        if self.stair_step_size is not None:
            out = self.operation(x, self.stair_step_size)
            return out

        if self.damp_gamma is not None:
            out = self.operation(x, self.damp_gamma, self.damp_p)
            return out

        # Otherwise: allow runtime kwargs (e.g. r) for surrogate ops
        if runtime_kwargs and self.fname in ("sin"):  # choose the correct name you use
            return self.operation(x, **runtime_kwargs)
        return self.operation(x)


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
                # model = UnarySurrogate(
                #     sin_operation, cfg, op, optype,
                #     damp_gamma=params["damp_gamma"], damp_p=params["damp_p"]
                # )
                model = UnarySurrogate(sin_operation_surrogate, cfg, op, optype)
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
