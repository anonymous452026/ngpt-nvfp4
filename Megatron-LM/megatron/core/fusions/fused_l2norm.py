"""Fused L2Norm (justnorm) Triton kernels for nGPT.

Fuses the 5-kernel L2 normalization (BF16→FP32 cast → pow → sum → rsqrt → mul → cast back)
into a single Triton kernel per row, eliminating intermediate global memory round-trips.

Usage:
    Set NVTE_JUSTNORM_MODE=triton to activate via the dispatcher in
    megatron/core/transformer/utils.py.
"""

from unittest.mock import MagicMock

import torch
from packaging import version

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    if version.parse(triton.__version__) < version.parse("3.4.0") and not torch.cuda.is_available():
        HAVE_TRITON = False
    else:
        HAVE_TRITON = tl.constexpr(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()


# ---------------------------------------------------------------------------
# L2Norm forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _l2norm_fwd_kernel(
    X_ptr,
    Y_ptr,
    RNORM_ptr,
    stride_x_row,
    stride_y_row,
    D: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Forward kernel: y = x * rsqrt(sum(x^2) + eps), one program per row."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x_ptrs = X_ptr + row * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    rnorm = 1.0 / tl.sqrt(sum_sq + eps)

    y = x * rnorm

    y_ptrs = Y_ptr + row * stride_y_row + cols
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(RNORM_ptr + row, rnorm)


# ---------------------------------------------------------------------------
# L2Norm backward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _l2norm_bwd_kernel(
    DY_ptr,
    Y_ptr,
    RNORM_ptr,
    DX_ptr,
    stride_dy_row,
    stride_y_row,
    stride_dx_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward kernel: dx = (dy - y * dot(dy, y)) * rnorm, one program per row."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + row * stride_y_row + cols, mask=mask, other=0.0).to(tl.float32)
    rnorm = tl.load(RNORM_ptr + row)

    dot_val = tl.sum(dy * y, axis=0)
    dx = (dy - y * dot_val) * rnorm

    tl.store(DX_ptr + row * stride_dx_row + cols, dx.to(DX_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# L2Norm autograd Function
# ---------------------------------------------------------------------------

class FusedL2Norm(torch.autograd.Function):
    """Autograd wrapper for the fused L2Norm Triton kernels."""

    @staticmethod
    def forward(ctx, x, eps):
        assert x.is_contiguous(), "fused_l2norm requires contiguous input on last dim"
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        N, D = x_2d.shape

        y = torch.empty_like(x_2d)
        rnorm = torch.empty(N, dtype=torch.float32, device=x.device)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (N,)

        _l2norm_fwd_kernel[grid](
            x_2d, y, rnorm,
            x_2d.stride(0), y.stride(0),
            D, eps,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(y, rnorm)
        ctx.orig_shape = orig_shape
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        y, rnorm = ctx.saved_tensors
        dy = dy.contiguous()
        dy_2d = dy.view(-1, ctx.D)
        N = dy_2d.shape[0]

        dx = torch.empty_like(dy_2d)
        grid = (N,)

        _l2norm_bwd_kernel[grid](
            dy_2d, y, rnorm, dx,
            dy_2d.stride(0), y.stride(0), dx.stride(0),
            ctx.D,
            BLOCK_D=ctx.BLOCK_D,
        )

        return dx.view(ctx.orig_shape), None


def fused_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Fused L2 normalization using Triton.

    Args:
        x: Input tensor.
        dim: Dimension to normalize over (must be -1 or last dim).
        eps: Small constant for numerical stability.

    Returns:
        L2-normalized tensor with same shape and dtype as input.
    """
    ndim = x.ndim
    if dim < 0:
        dim = ndim + dim

    if dim == ndim - 1:
        return FusedL2Norm.apply(x.contiguous(), eps)

    # For non-last dim, transpose so target dim is last, normalize, transpose back
    x = x.transpose(dim, -1).contiguous()
    y = FusedL2Norm.apply(x, eps)
    return y.transpose(dim, -1).contiguous()


# ---------------------------------------------------------------------------
# Fused SLERP + L2Norm forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _slerp_l2norm_fwd_kernel(
    A_ptr,
    B_ptr,
    LR_ptr,
    Y_ptr,
    stride_a_row,
    stride_b_row,
    stride_y_row,
    D: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Fused SLERP L2Norm forward: y = l2norm(l2norm(A) + lr * (l2norm(B) - l2norm(A))).

    O3 optimization: only outputs y, no intermediate tensors saved (recomputed in backward).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    a = tl.load(A_ptr + row * stride_a_row + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + row * stride_b_row + cols, mask=mask, other=0.0).to(tl.float32)
    lr = tl.load(LR_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # L2Norm(A)
    sum_sq_a = tl.sum(a * a, axis=0)
    rnorm_a = 1.0 / tl.sqrt(sum_sq_a + eps)
    ya = a * rnorm_a

    # L2Norm(B)
    sum_sq_b = tl.sum(b * b, axis=0)
    rnorm_b = 1.0 / tl.sqrt(sum_sq_b + eps)
    yb = b * rnorm_b

    # SLERP interpolation
    slerp = ya + lr * (yb - ya)

    # L2Norm(slerp)
    sum_sq_s = tl.sum(slerp * slerp, axis=0)
    rnorm_s = 1.0 / tl.sqrt(sum_sq_s + eps)
    y = slerp * rnorm_s

    # Store only the final output (intermediates recomputed in backward)
    tl.store(Y_ptr + row * stride_y_row + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Fused SLERP + L2Norm backward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _slerp_l2norm_bwd_recompute_kernel(
    DY_ptr,
    A_ptr,
    B_ptr,
    LR_ptr,
    DA_ptr,
    DB_ptr,
    DLR_ptr,
    stride_dy_row,
    stride_a_row,
    stride_b_row,
    stride_da_row,
    stride_db_row,
    D: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Fused SLERP L2Norm backward with forward recomputation (O3 optimization).

    Recomputes ya, yb, y, rnorm_a, rnorm_b, rnorm_s from A, B, lr instead of
    loading them from saved tensors. Saves 5 tensors (ya, yb, rnorm_a, rnorm_b, rnorm_s)
    from being stored during forward pass.

    Chain rule through: y = l2norm(slerp), slerp = ya + lr*(yb - ya),
    ya = l2norm(A), yb = l2norm(B).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_ptr + row * stride_a_row + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + row * stride_b_row + cols, mask=mask, other=0.0).to(tl.float32)
    lr = tl.load(LR_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # --- Recompute forward ---
    # L2Norm(A)
    sum_sq_a = tl.sum(a * a, axis=0)
    rnorm_a = 1.0 / tl.sqrt(sum_sq_a + eps)
    ya = a * rnorm_a

    # L2Norm(B)
    sum_sq_b = tl.sum(b * b, axis=0)
    rnorm_b = 1.0 / tl.sqrt(sum_sq_b + eps)
    yb = b * rnorm_b

    # SLERP interpolation
    slerp = ya + lr * (yb - ya)

    # L2Norm(slerp)
    sum_sq_s = tl.sum(slerp * slerp, axis=0)
    rnorm_s = 1.0 / tl.sqrt(sum_sq_s + eps)
    y = slerp * rnorm_s

    # --- Backward ---
    # Backward through l2norm(slerp): d_slerp = (dy - y * dot(dy, y)) * rnorm_s
    dot_dy_y = tl.sum(dy * y, axis=0)
    d_slerp = (dy - y * dot_dy_y) * rnorm_s

    # Gradient for lr: d_lr = d_slerp * (yb - ya), accumulated across rows
    d_lr = d_slerp * (yb - ya)
    tl.atomic_add(DLR_ptr + cols, d_lr, mask=mask)

    # Backward through slerp = ya + lr * (yb - ya):
    d_ya = d_slerp * (1.0 - lr)
    d_yb = d_slerp * lr

    # Backward through l2norm(A): da = (d_ya - ya * dot(d_ya, ya)) * rnorm_a
    dot_dya_ya = tl.sum(d_ya * ya, axis=0)
    da = (d_ya - ya * dot_dya_ya) * rnorm_a

    # Backward through l2norm(B): db = (d_yb - yb * dot(d_yb, yb)) * rnorm_b
    dot_dyb_yb = tl.sum(d_yb * yb, axis=0)
    db = (d_yb - yb * dot_dyb_yb) * rnorm_b

    tl.store(DA_ptr + row * stride_da_row + cols, da.to(DA_ptr.dtype.element_ty), mask=mask)
    tl.store(DB_ptr + row * stride_db_row + cols, db.to(DB_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Fused SLERP L2Norm autograd Function
# ---------------------------------------------------------------------------

class FusedSlerpL2Norm(torch.autograd.Function):
    """Autograd wrapper for the fused SLERP + L2Norm Triton kernels.

    O3 optimization: recomputes forward intermediates (ya, yb, rnorm_a, rnorm_b, rnorm_s)
    in backward instead of saving them, reducing memory by ~5 tensors per call.
    """

    @staticmethod
    def forward(ctx, A, B, lr, eps):
        assert A.is_contiguous() and B.is_contiguous()
        orig_shape = A.shape
        A_2d = A.view(-1, orig_shape[-1])
        B_2d = B.view(-1, orig_shape[-1])
        N, D = A_2d.shape

        y = torch.empty_like(A_2d)

        # Ensure lr is a tensor on the correct device (scalar or per-hidden-dim vector)
        if not isinstance(lr, torch.Tensor):
            lr_tensor = torch.tensor(lr, dtype=torch.float32, device=A.device)
        else:
            lr_tensor = lr.to(device=A.device)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (N,)

        _slerp_l2norm_fwd_kernel[grid](
            A_2d, B_2d, lr_tensor, y,
            A_2d.stride(0), B_2d.stride(0), y.stride(0),
            D, eps,
            BLOCK_D=BLOCK_D,
        )

        # O3: save only inputs (A, B, lr) for recomputation in backward
        ctx.save_for_backward(A, B, lr_tensor)
        ctx.orig_shape = orig_shape
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        ctx.eps = eps
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        A, B, lr_tensor = ctx.saved_tensors
        dy = dy.contiguous()
        dy_2d = dy.view(-1, ctx.D)
        A_2d = A.view(-1, ctx.D)
        B_2d = B.view(-1, ctx.D)
        N = dy_2d.shape[0]

        da = torch.empty_like(dy_2d)
        db = torch.empty_like(dy_2d)
        d_lr = torch.zeros(ctx.D, dtype=torch.float32, device=dy.device)
        grid = (N,)

        # O3: recompute forward intermediates inside the backward kernel
        _slerp_l2norm_bwd_recompute_kernel[grid](
            dy_2d, A_2d, B_2d, lr_tensor,
            da, db, d_lr,
            dy_2d.stride(0), A_2d.stride(0), B_2d.stride(0),
            da.stride(0), db.stride(0),
            ctx.D, ctx.eps,
            BLOCK_D=ctx.BLOCK_D,
        )

        return da.view(ctx.orig_shape), db.view(ctx.orig_shape), d_lr, None


def fused_slerp_l2norm(
    A: torch.Tensor,
    B: torch.Tensor,
    lr: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused SLERP + L2 normalization using Triton.

    Computes: l2norm(l2norm(A) + lr * (l2norm(B) - l2norm(A)))

    Args:
        A: First input tensor.
        B: Second input tensor (same shape as A).
        lr: Learning rate tensor (scalar or per-hidden-dim vector).
        dim: Dimension to normalize over (must be -1 or last dim).
        eps: Small constant for numerical stability.

    Returns:
        SLERP-interpolated and L2-normalized tensor.
    """
    ndim = A.ndim
    if dim < 0:
        dim = ndim + dim

    if dim == ndim - 1:
        return FusedSlerpL2Norm.apply(A.contiguous(), B.contiguous(), lr, eps)

    # For non-last dim, transpose so target dim is last, compute, transpose back
    A = A.transpose(dim, -1).contiguous()
    B = B.transpose(dim, -1).contiguous()
    y = FusedSlerpL2Norm.apply(A, B, lr, eps)
    return y.transpose(dim, -1).contiguous()


# ---------------------------------------------------------------------------
# Fused GeLU + L2Norm kernels
# ---------------------------------------------------------------------------
# Used by nGPT between the two FFN GEMMs: replaces the sequential
#   h = gelu(x)  →  y = l2norm(h)
# with a single Triton kernel, saving one global memory round-trip.
#
# GeLU uses the standard tanh approximation:
#   gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# implemented via exp for portability across Triton versions:
#   tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)

@triton.jit
def _gelu_l2norm_fwd_kernel(
    X_ptr,
    Y_ptr,
    RNORM_ptr,
    stride_x_row,
    stride_y_row,
    D: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Forward: y = l2norm(gelu(x)), one program per row.

    Saves rnorm and the original x for use in the backward pass.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)

    # GeLU (tanh approximation, stable via exp on clamped argument)
    coeff = 0.7978845608028654  # sqrt(2/pi)
    z = coeff * (x + 0.044715 * x * x * x)
    # clamp before exp to avoid overflow; tanh saturates well before ±10
    z = tl.where(z > 10.0, 10.0, tl.where(z < -10.0, -10.0, z))
    e2z = tl.exp(2.0 * z)
    tanh_z = (e2z - 1.0) / (e2z + 1.0)
    h = x * 0.5 * (1.0 + tanh_z)

    # L2Norm
    sum_sq = tl.sum(h * h, axis=0)
    rnorm = 1.0 / tl.sqrt(sum_sq + eps)
    y = h * rnorm

    tl.store(Y_ptr + row * stride_y_row + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(RNORM_ptr + row, rnorm)


@triton.jit
def _gelu_l2norm_bwd_kernel(
    DY_ptr,
    Y_ptr,
    X_ptr,
    RNORM_ptr,
    DX_ptr,
    stride_dy_row,
    stride_y_row,
    stride_x_row,
    stride_dx_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward: dx = gelu'(x) * l2norm_bwd(dy, y, rnorm), one program per row.

    Chain rule: dy → d_h (L2Norm backward) → dx (GeLU backward).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy    = tl.load(DY_ptr   + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    y     = tl.load(Y_ptr    + row * stride_y_row  + cols, mask=mask, other=0.0).to(tl.float32)
    x     = tl.load(X_ptr    + row * stride_x_row  + cols, mask=mask, other=0.0).to(tl.float32)
    rnorm = tl.load(RNORM_ptr + row)

    # 1) Backward through L2Norm: d_h = (dy - y * dot(dy, y)) * rnorm
    dot_val = tl.sum(dy * y, axis=0)
    d_h = (dy - y * dot_val) * rnorm

    # 2) Backward through GeLU: dx = d_h * gelu'(x)
    #    gelu'(x) = 0.5*(1+tanh(z)) + 0.5*x*(1-tanh(z)^2)*coeff*(1+3*0.044715*x^2)
    coeff = 0.7978845608028654
    z = coeff * (x + 0.044715 * x * x * x)
    z = tl.where(z > 10.0, 10.0, tl.where(z < -10.0, -10.0, z))
    e2z = tl.exp(2.0 * z)
    tanh_z = (e2z - 1.0) / (e2z + 1.0)
    gelu_prime = (
        0.5 * (1.0 + tanh_z)
        + 0.5 * x * (1.0 - tanh_z * tanh_z) * coeff * (1.0 + 3.0 * 0.044715 * x * x)
    )
    dx = d_h * gelu_prime

    tl.store(DX_ptr + row * stride_dx_row + cols, dx.to(DX_ptr.dtype.element_ty), mask=mask)


class FusedGeLUL2Norm(torch.autograd.Function):
    """Autograd wrapper for fused GeLU + L2Norm Triton kernels.

    Computes y = l2norm(gelu(x)) in a single kernel, eliminating the
    intermediate global memory write of the GeLU output.

    Saves x and rnorm for backward (not the intermediate gelu output h).
    """

    @staticmethod
    def forward(ctx, x, eps):
        assert x.is_contiguous()
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        N, D = x_2d.shape

        y     = torch.empty_like(x_2d)
        rnorm = torch.empty(N, dtype=torch.float32, device=x.device)

        BLOCK_D = triton.next_power_of_2(D)
        _gelu_l2norm_fwd_kernel[(N,)](
            x_2d, y, rnorm,
            x_2d.stride(0), y.stride(0),
            D, eps,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(y, x_2d, rnorm)
        ctx.orig_shape = orig_shape
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        y, x_2d, rnorm = ctx.saved_tensors
        dy = dy.contiguous().view(-1, ctx.D)
        N  = dy.shape[0]

        dx = torch.empty_like(dy)
        _gelu_l2norm_bwd_kernel[(N,)](
            dy, y, x_2d, rnorm, dx,
            dy.stride(0), y.stride(0), x_2d.stride(0), dx.stride(0),
            ctx.D,
            BLOCK_D=ctx.BLOCK_D,
        )
        return dx.view(ctx.orig_shape), None


def fused_gelu_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused GeLU + L2 normalization using Triton.

    Computes y = l2norm(gelu(x)) in a single kernel pass, eliminating the
    intermediate memory round-trip for the GeLU output.

    Args:
        x: Input tensor (last dim is the hidden dimension).
        eps: Numerical stability constant for L2Norm.

    Returns:
        L2-normalized GeLU output, same shape and dtype as x.
    """
    return FusedGeLUL2Norm.apply(x.contiguous(), eps)


# ---------------------------------------------------------------------------
# Scaled L2Norm forward kernel (Optimization #6)
# ---------------------------------------------------------------------------

@triton.jit
def _scaled_l2norm_fwd_kernel(
    X_ptr,
    SCALE_ptr,
    Y_ptr,
    RNORM_ptr,
    stride_x_row,
    stride_scale_row,
    stride_y_row,
    num_scale_rows,
    D: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Fused scaled L2Norm forward: y = scale^2 * l2norm(x), one program per row.

    Eliminates 3 separate kernel launches (l2norm, square, multiply) into one pass.

    num_scale_rows: number of rows in scale buffer. row % num_scale_rows cycles
    through scale groups (e.g. per-head scales broadcast across batch*seq rows).
    When stride_scale_row == 0, scale is 1D and broadcast regardless of num_scale_rows.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x_ptrs = X_ptr + row * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Use modulo so per-head scales cycle correctly across batch*seq rows.
    # When stride_scale_row == 0 (broadcast), row % 1 == 0 so offset is always 0.
    scale_row = row % num_scale_rows
    scale_ptrs = SCALE_ptr + scale_row * stride_scale_row + cols
    scale = tl.load(scale_ptrs, mask=mask, other=1.0).to(tl.float32)
    scale_sq = scale * scale

    # L2 normalization
    sum_sq = tl.sum(x * x, axis=0)
    rnorm = 1.0 / tl.sqrt(sum_sq + eps)
    x_norm = x * rnorm

    # Apply squared scale
    y = scale_sq * x_norm

    y_ptrs = Y_ptr + row * stride_y_row + cols
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(RNORM_ptr + row, rnorm)


@triton.jit
def _scaled_l2norm_bwd_kernel(
    DY_ptr,
    X_ptr,
    SCALE_ptr,
    RNORM_ptr,
    DX_ptr,
    DSCALE_ptr,
    stride_dy_row,
    stride_x_row,
    stride_scale_row,
    stride_dx_row,
    stride_dscale_row,
    num_scale_rows,
    D: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Backward kernel for scaled L2Norm.

    Chain rule: y = scale^2 * (x / ||x||)
    d_scale = 2 * scale * sum_rows(dy · x_norm)
    dx = scale^2 * (dy - x_norm * dot(dy, x_norm)) / ||x||
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    scale_row = row % num_scale_rows
    scale = tl.load(SCALE_ptr + scale_row * stride_scale_row + cols, mask=mask, other=1.0).to(tl.float32)
    rnorm = tl.load(RNORM_ptr + row)

    scale_sq = scale * scale
    x_norm = x * rnorm

    # Gradient w.r.t. normalized x: d_x_norm = dy * scale^2
    d_x_norm = dy * scale_sq

    # Gradient w.r.t. scale: atomic_add accumulates grads from all rows sharing this scale group.
    d_scale = 2.0 * scale * (dy * x_norm)
    dscale_ptrs = DSCALE_ptr + scale_row * stride_dscale_row + cols
    tl.atomic_add(dscale_ptrs, d_scale, mask=mask)

    # Gradient w.r.t. x through l2norm: dx = (d_x_norm - x_norm * dot(d_x_norm, x_norm)) * rnorm
    dot_val = tl.sum(d_x_norm * x_norm, axis=0)
    dx = (d_x_norm - x_norm * dot_val) * rnorm

    tl.store(DX_ptr + row * stride_dx_row + cols, dx.to(DX_ptr.dtype.element_ty), mask=mask)


class FusedScaledL2Norm(torch.autograd.Function):
    """Autograd wrapper for the fused scaled L2Norm Triton kernel (Optimization #6)."""

    @staticmethod
    def forward(ctx, x, scale, eps):
        assert x.is_contiguous(), "fused_scaled_l2norm requires contiguous input"
        orig_shape = x.shape
        orig_scale_shape = scale.shape if isinstance(scale, torch.Tensor) else None

        # Reshape x to 2D: [M, D] where D is last dim
        x_2d = x.view(-1, orig_shape[-1])
        N, D = x_2d.shape

        # **SIMPLIFIED APPROACH**: Flatten scale to 1D matching hidden_size
        # All rows will use a repeating pattern of this scale
        if not isinstance(scale, torch.Tensor):
            scale_1d = torch.full((D,), scale, dtype=torch.float32, device=x.device)
        else:
            scale = scale.to(device=x.device, dtype=torch.float32)
            # Flatten to 1D - total elements should be divisible by D
            scale_flat = scale.reshape(-1)

            if scale_flat.numel() == D:
                # Simple case: scale matches D exactly
                scale_1d = scale_flat
            else:
                # Scale has more elements (e.g., [1,1,16,128] = 2048 for D=128)
                # Ensure it's a multiple of D
                assert scale_flat.numel() % D == 0, f"Scale size {scale_flat.numel()} not multiple of D={D}"
                # Take first D elements (will cycle through in kernel if needed)
                # Actually, reshape to [num_groups, D] and we'll index by row
                num_groups = scale_flat.numel() // D
                scale_1d = scale_flat.view(num_groups, D)

        # Determine stride for scale access
        if scale_1d.ndim == 1:
            stride_scale_row = 0  # Broadcast to all rows
            ctx.num_scale_groups = 1
        else:
            stride_scale_row = scale_1d.stride(0)
            ctx.num_scale_groups = scale_1d.size(0)

        y = torch.empty_like(x_2d)
        rnorm = torch.empty(N, dtype=torch.float32, device=x.device)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (N,)

        _scaled_l2norm_fwd_kernel[grid](
            x_2d, scale_1d, y, rnorm,
            x_2d.stride(0), stride_scale_row, y.stride(0),
            ctx.num_scale_groups,
            D, eps,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_2d, scale_1d, rnorm)
        ctx.orig_shape = orig_shape
        ctx.orig_scale_shape = orig_scale_shape
        ctx.stride_scale_row = stride_scale_row
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        ctx.eps = eps
        ctx.N = N
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, scale_tensor, rnorm = ctx.saved_tensors
        dy = dy.contiguous()
        dy_2d = dy.view(-1, ctx.D)

        dx = torch.empty_like(dy_2d)

        # Create d_scale buffer matching scale's shape
        d_scale = torch.zeros_like(scale_tensor, dtype=torch.float32)
        if ctx.stride_scale_row == 0:
            stride_dscale_row = 0
        else:
            stride_dscale_row = d_scale.stride(0)

        grid = (ctx.N,)

        _scaled_l2norm_bwd_kernel[grid](
            dy_2d, x_2d, scale_tensor, rnorm, dx, d_scale,
            dy_2d.stride(0), x_2d.stride(0), ctx.stride_scale_row,
            dx.stride(0), stride_dscale_row,
            ctx.num_scale_groups,
            ctx.D, ctx.eps,
            BLOCK_D=ctx.BLOCK_D,
        )

        # Reshape d_scale back to original scale shape (handles scalar, 1D, and multi-group cases).
        # If scale was a Python scalar (not a tensor Variable), return None — PyTorch would raise
        # "returned a gradient different than None at position N, but the corresponding forward
        # input was not a Variable" if we return a tensor gradient for a non-tensor input.
        if ctx.orig_scale_shape is None:
            # scale was a Python float, not a trainable tensor — no gradient to return
            scale_grad = None
        else:
            scale_grad = d_scale.reshape(ctx.orig_scale_shape)

        return dx.view(ctx.orig_shape), scale_grad, None


def fused_scaled_l2norm(
    x: torch.Tensor,
    scale: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused scaled L2 normalization using Triton (Optimization #6).

    Computes: scale^2 * l2norm(x) in a single kernel pass.
    Eliminates 3 separate operations: l2norm(x), scale^2, multiply.

    Args:
        x: Input tensor.
        scale: Scale parameter (scalar or per-hidden-dim vector).
        dim: Dimension to normalize over (must be -1 or last dim).
        eps: Small constant for numerical stability.

    Returns:
        Scaled and L2-normalized tensor with same shape and dtype as input.
    """
    ndim = x.ndim
    if dim < 0:
        dim = ndim + dim

    if dim == ndim - 1:
        return FusedScaledL2Norm.apply(x.contiguous(), scale, eps)

    # For non-last dim, transpose so target dim is last, normalize, transpose back
    x = x.transpose(dim, -1).contiguous()
    y = FusedScaledL2Norm.apply(x, scale, eps)
    return y.transpose(dim, -1).contiguous()


# ---------------------------------------------------------------------------
# SLERP L2Norm with inline alpha computation (Optimization #7)
# ---------------------------------------------------------------------------

@triton.jit
def _slerp_l2norm_alpha_fwd_kernel(
    A_ptr,
    B_ptr,
    ALPHA_ptr,
    ALPHA_SCALE_ptr,
    Y_ptr,
    stride_a_row,
    stride_b_row,
    stride_y_row,
    N_B,
    D: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    """Fused SLERP L2Norm forward with inline lr = abs(alpha * alpha_scale) computation.

    O3+O7: Only outputs y, no intermediate tensors saved (recomputed in backward).
    Supports broadcasting: B can have fewer rows than A (B row = row % N_B).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    a = tl.load(A_ptr + row * stride_a_row + cols, mask=mask, other=0.0).to(tl.float32)
    b_row = row % N_B
    b = tl.load(B_ptr + b_row * stride_b_row + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute lr inline: lr = abs(alpha * alpha_scale)
    alpha = tl.load(ALPHA_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    alpha_scale = tl.load(ALPHA_SCALE_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    lr = tl.abs(alpha * alpha_scale)

    # L2Norm(A)
    sum_sq_a = tl.sum(a * a, axis=0)
    rnorm_a = 1.0 / tl.sqrt(sum_sq_a + eps)
    ya = a * rnorm_a

    # L2Norm(B)
    sum_sq_b = tl.sum(b * b, axis=0)
    rnorm_b = 1.0 / tl.sqrt(sum_sq_b + eps)
    yb = b * rnorm_b

    # SLERP interpolation
    slerp = ya + lr * (yb - ya)

    # L2Norm(slerp)
    sum_sq_s = tl.sum(slerp * slerp, axis=0)
    rnorm_s = 1.0 / tl.sqrt(sum_sq_s + eps)
    y = slerp * rnorm_s

    # Store only the final output (intermediates recomputed in backward)
    tl.store(Y_ptr + row * stride_y_row + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _slerp_l2norm_alpha_bwd_recompute_kernel(
    DY_ptr,
    A_ptr,
    B_ptr,
    ALPHA_ptr,
    ALPHA_SCALE_ptr,
    DA_ptr,
    DB_ptr,
    DALPHA_ptr,
    stride_dy_row,
    stride_a_row,
    stride_b_row,
    stride_da_row,
    stride_db_row,
    N_B,
    D: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
    BROADCAST_B: tl.constexpr,
):
    """Backward kernel for SLERP L2Norm with alpha, recomputing forward (O3 optimization).

    Recomputes ya, yb, y, rnorm_a, rnorm_b, rnorm_s from A, B, alpha, alpha_scale.
    Saves 6 tensors (ya, yb, y, rnorm_a, rnorm_b, rnorm_s) from being stored in forward.
    Supports broadcasting: when BROADCAST_B, db is accumulated with atomic_add.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(A_ptr + row * stride_a_row + cols, mask=mask, other=0.0).to(tl.float32)
    b_row = row % N_B
    b = tl.load(B_ptr + b_row * stride_b_row + cols, mask=mask, other=0.0).to(tl.float32)

    # Recompute lr
    alpha = tl.load(ALPHA_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    alpha_scale = tl.load(ALPHA_SCALE_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    alpha_scaled = alpha * alpha_scale
    lr = tl.abs(alpha_scaled)

    # --- Recompute forward ---
    # L2Norm(A)
    sum_sq_a = tl.sum(a * a, axis=0)
    rnorm_a = 1.0 / tl.sqrt(sum_sq_a + eps)
    ya = a * rnorm_a

    # L2Norm(B)
    sum_sq_b = tl.sum(b * b, axis=0)
    rnorm_b = 1.0 / tl.sqrt(sum_sq_b + eps)
    yb = b * rnorm_b

    # SLERP interpolation
    slerp = ya + lr * (yb - ya)

    # L2Norm(slerp)
    sum_sq_s = tl.sum(slerp * slerp, axis=0)
    rnorm_s = 1.0 / tl.sqrt(sum_sq_s + eps)
    y = slerp * rnorm_s

    # --- Backward ---
    # Backward through l2norm(slerp)
    dot_dy_y = tl.sum(dy * y, axis=0)
    d_slerp = (dy - y * dot_dy_y) * rnorm_s

    # Gradient for lr: d_lr = d_slerp * (yb - ya)
    d_lr = d_slerp * (yb - ya)

    # Gradient through abs: d(abs(x))/dx = sign(x)
    sign_alpha = tl.where(alpha_scaled >= 0.0, 1.0, -1.0)
    d_alpha = d_lr * sign_alpha * alpha_scale
    tl.atomic_add(DALPHA_ptr + cols, d_alpha, mask=mask)

    # Backward through slerp = ya + lr * (yb - ya):
    d_ya = d_slerp * (1.0 - lr)
    d_yb = d_slerp * lr

    # Backward through l2norm(A)
    dot_dya_ya = tl.sum(d_ya * ya, axis=0)
    da = (d_ya - ya * dot_dya_ya) * rnorm_a

    # Backward through l2norm(B)
    dot_dyb_yb = tl.sum(d_yb * yb, axis=0)
    db = (d_yb - yb * dot_dyb_yb) * rnorm_b

    tl.store(DA_ptr + row * stride_da_row + cols, da.to(DA_ptr.dtype.element_ty), mask=mask)
    if BROADCAST_B:
        b_row2 = row % N_B
        tl.atomic_add(DB_ptr + b_row2 * stride_db_row + cols, db, mask=mask)
    else:
        tl.store(DB_ptr + row * stride_db_row + cols, db.to(DB_ptr.dtype.element_ty), mask=mask)


class FusedSlerpL2NormAlpha(torch.autograd.Function):
    """Autograd wrapper for SLERP L2Norm with inline alpha computation (O7+O3).

    O3 optimization: recomputes forward intermediates in backward, saving 6 tensors.
    """

    @staticmethod
    def forward(ctx, A, B, alpha_param, alpha_scale, eps):
        assert A.is_contiguous() and B.is_contiguous()
        orig_shape = A.shape
        orig_B_shape = B.shape
        A_2d = A.view(-1, orig_shape[-1])
        B_2d = B.view(-1, orig_shape[-1])
        N, D = A_2d.shape
        N_B = B_2d.shape[0]
        broadcast_B = (N_B != N)

        y = torch.empty_like(A_2d)

        # Save original shapes for backward pass
        orig_alpha_param_shape = alpha_param.shape if isinstance(alpha_param, torch.Tensor) else None

        # Ensure alpha_param and alpha_scale are 1D tensors of size D on the correct device.
        if not isinstance(alpha_param, torch.Tensor):
            alpha_param = torch.full((D,), float(alpha_param), dtype=torch.float32, device=A.device)
        elif alpha_param.numel() == 1:
            alpha_param = alpha_param.expand(D).contiguous()
        else:
            alpha_param = alpha_param.to(device=A.device, dtype=torch.float32)

        if not isinstance(alpha_scale, torch.Tensor):
            alpha_scale = torch.full((D,), float(alpha_scale), dtype=torch.float32, device=A.device)
        elif alpha_scale.numel() == 1:
            alpha_scale = alpha_scale.expand(D).contiguous()
        else:
            alpha_scale = alpha_scale.to(device=A.device, dtype=torch.float32)

        BLOCK_D = triton.next_power_of_2(D)
        grid = (N,)

        _slerp_l2norm_alpha_fwd_kernel[grid](
            A_2d, B_2d, alpha_param, alpha_scale, y,
            A_2d.stride(0), B_2d.stride(0), y.stride(0),
            N_B,
            D, eps,
            BLOCK_D=BLOCK_D,
        )

        # O3: save only inputs for recomputation in backward
        ctx.save_for_backward(A, B, alpha_param, alpha_scale)
        ctx.orig_shape = orig_shape
        ctx.orig_B_shape = orig_B_shape
        ctx.orig_alpha_param_shape = orig_alpha_param_shape
        ctx.broadcast_B = broadcast_B
        ctx.N_B = N_B
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        ctx.eps = eps
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        A, B, alpha_param, alpha_scale = ctx.saved_tensors
        dy = dy.contiguous()
        dy_2d = dy.view(-1, ctx.D)
        A_2d = A.view(-1, ctx.D)
        B_2d = B.view(-1, ctx.D)
        N = dy_2d.shape[0]

        da = torch.empty_like(dy_2d)
        if ctx.broadcast_B:
            db = torch.zeros(ctx.N_B, ctx.D, dtype=torch.float32, device=dy.device)
            db_stride = ctx.D
        else:
            db = torch.empty_like(dy_2d)
            db_stride = db.stride(0)
        d_alpha = torch.zeros(ctx.D, dtype=torch.float32, device=dy.device)
        grid = (N,)

        # O3: recompute forward intermediates inside the backward kernel
        _slerp_l2norm_alpha_bwd_recompute_kernel[grid](
            dy_2d, A_2d, B_2d,
            alpha_param, alpha_scale,
            da, db, d_alpha,
            dy_2d.stride(0), A_2d.stride(0), B_2d.stride(0),
            da.stride(0), db_stride,
            ctx.N_B,
            ctx.D, ctx.eps,
            BLOCK_D=ctx.BLOCK_D,
            BROADCAST_B=ctx.broadcast_B,
        )

        # Reshape d_alpha to match original alpha_param shape
        if ctx.orig_alpha_param_shape is not None:
            d_alpha = d_alpha.view(ctx.orig_alpha_param_shape)

        # Convert db back to input dtype and reshape to original B shape
        if ctx.broadcast_B:
            db = db.to(dy.dtype).view(ctx.orig_B_shape)
        else:
            db = db.view(ctx.orig_shape)

        return da.view(ctx.orig_shape), db, d_alpha, None, None


def fused_slerp_l2norm_with_alpha(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha_param: torch.Tensor,
    alpha_scale: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused SLERP + L2 normalization with inline alpha computation (Optimization #7).

    Computes: l2norm(l2norm(A) + abs(alpha * alpha_scale) * (l2norm(B) - l2norm(A)))

    Eliminates 2 separate kernel launches (multiply + abs) by computing lr inside the kernel.

    Args:
        A: First input tensor.
        B: Second input tensor (same shape as A).
        alpha_param: Alpha parameter tensor (scalar or per-hidden-dim vector).
        alpha_scale: Precomputed alpha scaling factor (from __init__).
        dim: Dimension to normalize over (must be -1 or last dim).
        eps: Small constant for numerical stability.

    Returns:
        SLERP-interpolated and L2-normalized tensor.
    """
    ndim = A.ndim
    if dim < 0:
        dim = ndim + dim

    if dim == ndim - 1:
        return FusedSlerpL2NormAlpha.apply(A.contiguous(), B.contiguous(), alpha_param, alpha_scale, eps)

    # For non-last dim, transpose so target dim is last, compute, transpose back
    A = A.transpose(dim, -1).contiguous()
    B = B.transpose(dim, -1).contiguous()
    y = FusedSlerpL2NormAlpha.apply(A, B, alpha_param, alpha_scale, eps)
    return y.transpose(dim, -1).contiguous()


# ---------------------------------------------------------------------------
# Distributed L2Norm Triton kernels (for tensor-parallel activations)
# ---------------------------------------------------------------------------
# When TP>1, the hidden dimension is split across ranks. We need:
#   Forward:  local_ss → all-reduce → normalize with global rnorm
#   Backward: local_dot(dy,y) → all-reduce → gradient with global dot

@triton.jit
def _l2norm_partial_ss_kernel(
    X_ptr,
    SS_ptr,
    stride_x_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute per-row local sum of squares: ss[row] = sum(x[row, :]^2).

    Fuses BF16→FP32 cast + square + reduction into a single pass.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    tl.store(SS_ptr + row, ss)


@triton.jit
def _l2norm_normalize_with_ss_kernel(
    X_ptr,
    SS_ptr,
    Y_ptr,
    RNORM_ptr,
    stride_x_row,
    stride_y_row,
    D: tl.constexpr,
    eps,
    BLOCK_D: tl.constexpr,
):
    """O4: Normalize using global sum-of-squares, computing rnorm inline.

    Fuses the separate PyTorch rnorm computation (sqrt + reciprocal) into the
    normalize kernel, eliminating one kernel launch.
    y[row] = x[row] * (1 / sqrt(ss[row] + eps))
    Also stores rnorm for use in backward pass.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    ss = tl.load(SS_ptr + row)
    rnorm = 1.0 / tl.sqrt(ss + eps)
    y = x * rnorm

    tl.store(Y_ptr + row * stride_y_row + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)
    tl.store(RNORM_ptr + row, rnorm)


@triton.jit
def _l2norm_bwd_dot_kernel(
    DY_ptr,
    Y_ptr,
    DOT_ptr,
    stride_dy_row,
    stride_y_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute per-row local dot product: dot[row] = sum(dy[row] * y[row])."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + row * stride_y_row + cols, mask=mask, other=0.0).to(tl.float32)
    dot_val = tl.sum(dy * y, axis=0)
    tl.store(DOT_ptr + row, dot_val)


@triton.jit
def _l2norm_bwd_grad_kernel(
    DY_ptr,
    Y_ptr,
    DOT_ptr,
    RNORM_ptr,
    DX_ptr,
    stride_dy_row,
    stride_y_row,
    stride_dx_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute gradient: dx = (dy - y * global_dot) * rnorm."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + row * stride_y_row + cols, mask=mask, other=0.0).to(tl.float32)
    dot_val = tl.load(DOT_ptr + row)
    rnorm = tl.load(RNORM_ptr + row)
    dx = (dy - y * dot_val) * rnorm

    tl.store(DX_ptr + row * stride_dx_row + cols, dx.to(DX_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Distributed L2Norm autograd Function
# ---------------------------------------------------------------------------

class DistributedFusedL2Norm(torch.autograd.Function):
    """Autograd wrapper for distributed (TP-aware) fused L2Norm.

    Forward:
        1. Triton kernel: compute local sum-of-squares per row
        2. All-reduce local SS across TP group → global SS
        3. Triton kernel: normalize with global rnorm

    Backward:
        1. Triton kernel: compute local dot(dy, y) per row
        2. All-reduce local dot across TP group → global dot
        3. Triton kernel: dx = (dy - y * global_dot) * rnorm
    """

    @staticmethod
    def forward(ctx, x, eps, tp_group):
        assert x.is_contiguous(), "distributed fused_l2norm requires contiguous input"
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        N, D = x_2d.shape

        BLOCK_D = triton.next_power_of_2(D)
        grid = (N,)

        # Step 1: local sum-of-squares (Triton)
        local_ss = torch.empty(N, dtype=torch.float32, device=x.device)
        _l2norm_partial_ss_kernel[grid](
            x_2d, local_ss, x_2d.stride(0), D, BLOCK_D=BLOCK_D,
        )

        # Step 2: all-reduce across TP
        torch.distributed.all_reduce(local_ss, group=tp_group)

        # Step 3: O4 — compute rnorm inline and normalize in one kernel
        y = torch.empty_like(x_2d)
        rnorm = torch.empty(N, dtype=torch.float32, device=x.device)
        _l2norm_normalize_with_ss_kernel[grid](
            x_2d, local_ss, y, rnorm,
            x_2d.stride(0), y.stride(0), D, eps, BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(y.view(orig_shape), rnorm)
        ctx.orig_shape = orig_shape
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        ctx.tp_group = tp_group
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        y, rnorm = ctx.saved_tensors
        dy = dy.contiguous()
        dy_2d = dy.view(-1, ctx.D)
        y_2d = y.view(-1, ctx.D)
        N = dy_2d.shape[0]
        grid = (N,)

        # Step 1: local dot(dy, y) (Triton)
        local_dot = torch.empty(N, dtype=torch.float32, device=dy.device)
        _l2norm_bwd_dot_kernel[grid](
            dy_2d, y_2d, local_dot,
            dy_2d.stride(0), y_2d.stride(0),
            ctx.D, BLOCK_D=ctx.BLOCK_D,
        )

        # Step 2: all-reduce dot across TP
        torch.distributed.all_reduce(local_dot, group=ctx.tp_group)

        # Step 3: compute gradient (Triton)
        dx = torch.empty_like(dy_2d)
        _l2norm_bwd_grad_kernel[grid](
            dy_2d, y_2d, local_dot, rnorm, dx,
            dy_2d.stride(0), y_2d.stride(0), dx.stride(0),
            ctx.D, BLOCK_D=ctx.BLOCK_D,
        )

        return dx.view(ctx.orig_shape), None, None


def fused_distributed_l2norm(
    x: torch.Tensor,
    tp_group,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused distributed L2 normalization using Triton with TP all-reduce.

    For tensor-parallel activations where the hidden dim is split across TP ranks.
    Uses Triton kernels for local computation (fused cast+square+reduce and
    fused normalize+cast-back) with all-reduce for global norm.

    Args:
        x: Input tensor (TP-local partition).
        tp_group: Tensor-parallel process group for all-reduce.
        dim: Dimension to normalize over (must be -1 or last dim).
        eps: Small constant for numerical stability.

    Returns:
        L2-normalized tensor using the global norm across all TP ranks.
    """
    ndim = x.ndim
    if dim < 0:
        dim = ndim + dim

    if dim == ndim - 1:
        return DistributedFusedL2Norm.apply(x.contiguous(), eps, tp_group)

    # For non-last dim, transpose so target dim is last, normalize, transpose back
    x = x.transpose(dim, -1).contiguous()
    y = DistributedFusedL2Norm.apply(x, eps, tp_group)
    return y.transpose(dim, -1).contiguous()


# ---------------------------------------------------------------------------
# Fused SUV Scale kernel (P2 optimization)
# ---------------------------------------------------------------------------
# Fuses: y = (suv_param * scale_factor) * x
# into a single Triton kernel, eliminating 2 separate PyTorch elementwise ops
# (param * scalar multiply + broadcast multiply) and the intermediate fp32 tensor.

@triton.jit
def _suv_scale_fwd_kernel(
    X_ptr,
    SUV_ptr,
    Y_ptr,
    scale_factor,
    stride_x_row,
    stride_y_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward: y[row, col] = suv[col] * scale_factor * x[row, col].

    Reads x in input dtype (bf16), suv in fp32, computes in fp32,
    writes y in input dtype (bf16). One kernel replaces two PyTorch ops.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    suv = tl.load(SUV_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (suv * scale_factor) * x

    tl.store(Y_ptr + row * stride_y_row + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _suv_scale_bwd_dx_kernel(
    DY_ptr,
    SUV_ptr,
    DX_ptr,
    scale_factor,
    stride_dy_row,
    stride_dx_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward dx: dx[row, col] = suv[col] * scale_factor * dy[row, col]."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    suv = tl.load(SUV_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dx = (suv * scale_factor) * dy

    tl.store(DX_ptr + row * stride_dx_row + cols, dx.to(DX_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _suv_scale_bwd_dsuv_kernel(
    DY_ptr,
    X_ptr,
    DSUV_ptr,
    scale_factor,
    stride_dy_row,
    stride_x_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward d_suv: accumulate scale_factor * dy[row] * x[row] into d_suv via atomic_add.

    One program per row. Each program computes its contribution to d_suv[col]
    and atomically adds it. This avoids creating a large intermediate tensor.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    contrib = scale_factor * dy * x

    tl.atomic_add(DSUV_ptr + cols, contrib, mask=mask)


class FusedSUVScale(torch.autograd.Function):
    """Fused SUV scaling: y = (suv_param * scale_factor) * x.

    Replaces 2 PyTorch elementwise ops (param*scalar + broadcast*activation)
    with a single Triton kernel in forward and 2 Triton kernels in backward.
    Also eliminates the intermediate fp32 tensor from PyTorch's type promotion.
    """

    @staticmethod
    def forward(ctx, x, suv_param, scale_factor):
        assert x.is_contiguous(), "fused_suv_scale requires contiguous input"
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        N, D = x_2d.shape

        suv_1d = suv_param.to(dtype=torch.float32, device=x.device)
        assert suv_1d.ndim == 1 and suv_1d.shape[0] == D, (
            f"SUV param must be 1D with size {D}, got shape {suv_1d.shape}"
        )

        y = torch.empty_like(x_2d)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (N,)

        _suv_scale_fwd_kernel[grid](
            x_2d, suv_1d, y,
            scale_factor,
            x_2d.stride(0), y.stride(0),
            D, BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_2d, suv_1d)
        ctx.scale_factor = scale_factor
        ctx.orig_shape = orig_shape
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        ctx.N = N
        return y.view(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, suv_1d = ctx.saved_tensors
        dy = dy.contiguous()
        dy_2d = dy.view(-1, ctx.D)
        N = ctx.N
        grid = (N,)

        # dx = suv * scale_factor * dy
        dx = torch.empty_like(dy_2d)
        _suv_scale_bwd_dx_kernel[grid](
            dy_2d, suv_1d, dx,
            ctx.scale_factor,
            dy_2d.stride(0), dx.stride(0),
            ctx.D, BLOCK_D=ctx.BLOCK_D,
        )

        # d_suv = scale_factor * sum_over_rows(dy * x), via atomic accumulation
        d_suv = torch.zeros_like(suv_1d)
        _suv_scale_bwd_dsuv_kernel[grid](
            dy_2d, x_2d, d_suv,
            ctx.scale_factor,
            dy_2d.stride(0), x_2d.stride(0),
            ctx.D, BLOCK_D=ctx.BLOCK_D,
        )

        return dx.view(ctx.orig_shape), d_suv, None


def fused_suv_scale(
    x: torch.Tensor,
    suv_param: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """Fused SUV scaling using Triton: y = (suv_param * scale_factor) * x.

    Replaces the two-step PyTorch computation:
        suv = suv_param * scale_factor   # elementwise on [D] param
        y = suv * x                      # broadcast multiply on [S, B, D] activation

    with a single Triton kernel that computes in fp32 and writes bf16 output,
    eliminating the intermediate fp32 activation tensor.

    Args:
        x: Input activation tensor (any shape, last dim = D).
        suv_param: SUV parameter tensor, 1D of size D (trainable).
        scale_factor: Scalar scaling factor (precomputed constant).

    Returns:
        Scaled tensor with same shape and dtype as x.
    """
    return FusedSUVScale.apply(x.contiguous(), suv_param, scale_factor)
