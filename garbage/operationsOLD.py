from __future__ import annotations

from typing import List
import torch
import torch.nn as nn


# def _safe_complex_log(z: torch.Tensor) -> torch.Tensor:
#     eps = 1e-6
#     max_r = 1e6

#     if torch.is_complex(z):
#         mag = z.abs()
#         mag_clamped = torch.clamp(mag, eps, max_r)
#         scale = mag_clamped / (mag + eps)
#         z_clamped = z * scale
#     else:
#         z_clamped = torch.clamp(z, eps, max_r)

#     out = torch.log(z_clamped)

#     finite_mask = torch.isfinite(out)
#     if not finite_mask.all():
#         out = out.clone()
#         out[~finite_mask] = 0.0
#     return out

# ======================
# UNARY OPERATIONS
# ======================

def identity_operation(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1)  # (B,) → (B,1)


def const_operation(x: torch.Tensor) -> torch.Tensor:
    return (x * 0 + 1).unsqueeze(-1)


def square_operation(x: torch.Tensor) -> torch.Tensor:
    return (x * x).unsqueeze(-1)


def cube_operation(x: torch.Tensor) -> torch.Tensor:
    return (x * x * x).unsqueeze(-1)


def sqrt_operation(x: torch.Tensor) -> torch.Tensor:
    # complex sqrt
    return torch.sqrt(x).unsqueeze(-1)


# def log_operation(x: torch.Tensor) -> torch.Tensor:
#     # complex logarithm
#     return torch.log(x).unsqueeze(-1)

def log_operation(x: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    """
    Complex logarithm with stair gating.

    For |x| <= stair_step_size / 2:
        output = 0
    For |x| > stair_step_size / 2:
        output = log(x_safe)

    Returns (B, 1)
    """
    eps = 1e-8

    z = x

    # magnitude
    mag = z.abs() if torch.is_complex(z) else torch.abs(z)

    # stair gate
    stair = torch.as_tensor(
        stair_step_size,
        device=z.device,
        dtype=mag.dtype,
    )
    mask = mag > (stair / 2.0)

    # avoid log(0) while preserving phase
    scale = mag / (mag + eps)
    z_safe = z * scale + eps

    out = torch.log(z_safe)

    # apply stair
    out = torch.where(mask, out, torch.zeros_like(out))

    # safety: remove NaN / Inf
    finite = torch.isfinite(out)
    if not finite.all():
        out = out.clone()
        out[~finite] = 0.0

    return out.unsqueeze(-1)


# def exponent_operation(x: torch.Tensor) -> torch.Tensor:
#     return torch.exp(x).unsqueeze(-1)
def exponent_operation(x: torch.Tensor) -> torch.Tensor:
    z = x
    if torch.is_complex(z):
        u = torch.clamp(z.real, -40.0, 40.0)  # 40 is safe for float32
        z = torch.complex(u, z.imag)
    else:
        z = torch.clamp(z, -40.0, 40.0)
    out = torch.exp(z)
    # out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.unsqueeze(-1)


def sin_operation(x: torch.Tensor, damp_gamma: float, damp_p: float) -> torch.Tensor:
    u = x.real
    v = x.imag
    eps = 1e-12
    damp = torch.exp(-damp_gamma * torch.pow(v.abs() + eps, damp_p))
    y_real = torch.sin(u) * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return y.unsqueeze(-1)


def cos_operation(x: torch.Tensor, damp_gamma: float, damp_p: float) -> torch.Tensor:
    u = x.real
    v = x.imag
    eps = 1e-12
    damp = torch.exp(-damp_gamma * torch.pow(v.abs() + eps, damp_p))
    y_real = torch.cos(u) * damp
    y = torch.complex(y_real, torch.zeros_like(y_real))
    return y.unsqueeze(-1)

def tan_operation(x: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    z = x

    # Prevent cosh(Im(z)) overflow inside sin/cos for complex inputs
    if torch.is_complex(z):
        v = torch.clamp(z.imag, -40.0, 40.0)   # 40 is safe-ish for float32
        z = torch.complex(z.real, v)

    den = torch.cos(z)
    den_mag = den.abs() if torch.is_complex(den) else torch.abs(den)

    stair = torch.as_tensor(stair_step_size, device=z.device, dtype=den_mag.dtype)
    mask = (den_mag > (stair / 2.0)) & torch.isfinite(den_mag)

    out = torch.zeros_like(z)
    if mask.any():
        num = torch.sin(z[mask])
        den_m = den[mask]
        out[mask] = num / den_m

    finite = torch.isfinite(out)
    if not finite.all():
        out = out.clone()
        out[~finite] = 0.0

    return out.unsqueeze(-1)

# def tanh_operation(x: torch.Tensor) -> torch.Tensor:
#     z = x
#     out = torch.tanh(z)

#     # safety: remove NaN / Inf (should be rare, but keep consistent)
#     finite = torch.isfinite(out)
#     if not finite.all():
#         out = out.clone()
#         out[~finite] = 0.0

#     return out.unsqueeze(-1)
def tanh_operation(x: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    z = x

    # Prevent cosh(Re(z)) overflow
    if torch.is_complex(z):
        u = torch.clamp(z.real, -40.0, 40.0)
        z = torch.complex(u, z.imag)
    else:
        z = torch.clamp(z, -40.0, 40.0)

    # Gate near poles: tanh(z) has poles where cosh(z)=0
    den = torch.cosh(z)
    den_mag = den.abs() if torch.is_complex(den) else torch.abs(den)

    stair = torch.as_tensor(stair_step_size, device=z.device, dtype=den_mag.dtype)
    mask = (den_mag > (stair / 2.0)) & torch.isfinite(den_mag)

    out = torch.zeros_like(z)
    if mask.any():
        out[mask] = torch.tanh(z[mask])  # stable

    finite = torch.isfinite(out)
    if not finite.all():
        out = out.clone()
        out[~finite] = 0.0

    return out.unsqueeze(-1)


# ======================
# BINARY OPERATIONS
# ======================

def multiplication_operation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return (x1 * x2).unsqueeze(-1)


# def div_operation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#     # plain complex division
#     return (x1 / x2).unsqueeze(-1)

def div_operation(x1: torch.Tensor, x2: torch.Tensor, stair_step_size: float) -> torch.Tensor:
    num, den = x1, x2

    stair = torch.as_tensor(
        stair_step_size,
        device=den.device,
        dtype=den.real.dtype if torch.is_complex(den) else den.dtype,
    )

    den_mag = den.abs() if torch.is_complex(den) else torch.abs(den)
    mask = den_mag > (stair / 2.0)

    out = torch.zeros_like(num)

    if mask.any():
        out[mask] = num[mask] / den[mask]

    return out.unsqueeze(-1)


# def power_operation(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#     return (x1 ** x2).unsqueeze(-1)

def power_operation(x1: torch.Tensor, x2: torch.Tensor, stair_step_size: float | None = None) -> torch.Tensor:
    """
    Complex power with optional stair on the base magnitude (via log|base|).

    - Always clamps exponent to avoid crazy growth.
    - If stair_step_size is given, we only keep outputs where |log|base|| <= stair_step_size.
      Outside that region, output 0.
    - If result is still non-finite and base ~ 0, set to 0.
    """
    base = x1
    exp = x2

    # clamp exponent to avoid insane magnitudes
    if torch.is_complex(exp):
        exp_real = torch.clamp(exp.real, -5.0, 5.0)
        exp_imag = torch.clamp(exp.imag, -5.0, 5.0)
        exp_safe = torch.complex(exp_real, exp_imag)
    else:
        exp_safe = torch.clamp(exp, -5.0, 5.0)

    out = base ** exp_safe  # complex

    # optional stair gating based on |log|base||
    if stair_step_size is not None:
        eps = 1e-8
        stair = torch.as_tensor(
            stair_step_size,
            device=base.device,
            dtype=base.real.dtype if torch.is_complex(base) else base.dtype,
        )

        base_mag = base.abs().clamp(min=eps)
        log_base = torch.log(base_mag)  # real
        mask = torch.abs(log_base) <= stair

        out_masked = torch.zeros_like(out)
        if mask.any():
            out_masked[mask] = out[mask]
        out = out_masked

    # fix remaining non-finite values
    finite_mask = torch.isfinite(out)
    if not finite_mask.all():
        out = out.clone()
        eps = 1e-6
        base_small = base.abs() < eps
        zero_fix = (~finite_mask) & base_small
        if zero_fix.any():
            out[zero_fix] = 0.0
        still_bad = ~torch.isfinite(out)
        if still_bad.any():
            out[still_bad] = 0.0

    return out.unsqueeze(-1)



# class UnarySurrogate(nn.Module):
#     """Wrap an exact unary operation (no learnable params)."""
#     def __init__(self, operation, cfg, fname, ftype):
#         super().__init__()
#         self.operation = operation
#         self.cfg = cfg
#         self.fname = fname
#         self.ftype = ftype

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B,)
#         out = self.operation(x)
#         return out  # (B,1)

class UnarySurrogate(nn.Module):
    """Wrap an exact unary operation (no learnable params here)."""
    def __init__(self, operation, cfg, fname, ftype, stair_step_size: float | None = None, damp_gamma: float | None = None, damp_p: float | None = None,):
        super().__init__()
        self.operation = operation
        self.cfg = cfg
        self.fname = fname
        self.ftype = ftype
        self.stair_step_size = stair_step_size
        self.damp_gamma = damp_gamma
        self.damp_p = damp_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,)
        if self.stair_step_size is not None:
            out = self.operation(x, self.stair_step_size)
        elif self.damp_gamma is not None:
            out = self.operation(x, self.damp_gamma, self.damp_p)
        else:
            out = self.operation(x)
        return out  # (B,1)

# class BinarySurrogate(nn.Module):
#     """Wrap an exact binary operation."""
#     def __init__(self, operation, cfg, fname, ftype):
#         super().__init__()
#         self.operation = operation
#         self.cfg = cfg
#         self.fname = fname
#         self.ftype = ftype

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         # X: (B, 2)
#         out = self.operation(X[:, 0], X[:, 1])
#         return out  # (B,1)

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
        # X: (B, 2)
        if self.stair_step_size is not None:
            out = self.operation(X[:, 0], X[:, 1], self.stair_step_size)
        else:
            out = self.operation(X[:, 0], X[:, 1])
        return out  # (B,1)

def load_models(cfg, layer_idx: int):
    """
    Build the list of exact symbolic operations for a given symbolic layer.
    """
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
                # model = UnarySurrogate(log_operation, cfg, op, optype)
                model = UnarySurrogate(log_operation, cfg, op, optype, stair_step_size=stair_step_size)
            elif op == "exp":
                model = UnarySurrogate(exponent_operation, cfg, op, optype)
            elif op == "sin":
                model = UnarySurrogate(sin_operation, cfg, op, optype, damp_gamma=params["damp_gamma"], damp_p=params["damp_p"])
            elif op == "cos":
                model = UnarySurrogate(cos_operation, cfg, op, optype, damp_gamma=params["damp_gamma"], damp_p=params["damp_p"])
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
                # model = BinarySurrogate(div_operation, cfg, op, optype)
                model = BinarySurrogate(div_operation, cfg, op, optype, stair_step_size=stair_step_size)
            elif op == "pow":
                # model = BinarySurrogate(power_operation, cfg, op, optype)
                model = BinarySurrogate(power_operation, cfg, op, optype, stair_step_size=stair_step_size)
            else:
                raise ValueError(f"Unknown exact binary operation '{op}'")
            binary_nos.append(model)
        else:
            raise ValueError(f"Unknown library_function_type '{optype}'")

    return unary_nos, binary_nos