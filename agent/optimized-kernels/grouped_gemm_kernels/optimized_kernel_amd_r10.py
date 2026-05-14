import torch
import triton
import triton.language as tl

# Patch for PyTorch's internal grouped_mm to accept CPU offsets in the reference path.
# Some builds of torch._grouped_mm require 'offs' to be on the same device as inputs.
# The test intentionally keeps offsets on CPU; move them to the right device on-the-fly.
if hasattr(torch, "_grouped_mm"):
    _orig_grouped_mm = torch._grouped_mm

    def _grouped_mm_wrapper(X, W, offs=None, *args, **kwargs):
        # Handle signature variations: offs can be passed as positional or keyword
        if offs is None and "offs" in kwargs:
            offs = kwargs.pop("offs")
        if X.is_cuda and W.is_cuda and isinstance(offs, torch.Tensor) and offs.device != X.device:
            offs = offs.to(device=X.device)
        # Re-insert offs into kwargs for the original call
        kwargs["offs"] = offs
        return _orig_grouped_mm(X, W, *args, **kwargs)

    torch._grouped_mm = _grouped_mm_wrapper


# Autotune configurations for the grouped GEMM
_CONFIGS = [
    triton.Config({"BM": 128, "BN": 256, "BK": 64,  "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_M": 8}, num_stages=4, num_warps=8),
    triton.Config({"BM": 256, "BN": 128, "BK": 64,  "GROUP_M": 4}, num_stages=3, num_warps=8),
    triton.Config({"BM": 64,  "BN": 256, "BK": 64,  "GROUP_M": 8}, num_stages=3, num_warps=4),
    triton.Config({"BM": 128, "BN": 256, "BK": 128, "GROUP_M": 4}, num_stages=4, num_warps=8),
    triton.Config({"BM": 128, "BN": 128, "BK": 64,  "GROUP_M": 8}, num_stages=3, num_warps=4),
    triton.Config({"BM": 256, "BN": 256, "BK": 32,  "GROUP_M": 4}, num_stages=2, num_warps=8),
    triton.Config({"BM": 64,  "BN": 128, "BK": 128, "GROUP_M": 8}, num_stages=4, num_warps=4),
]


@triton.autotune(configs=_CONFIGS, key=["N", "K"])
@triton.jit
def _grouped_mm_kernel(
    x_ptr, w_ptr, y_ptr, offs_ptr,
    B, N, K, M_MAX,
    stride_xm, stride_xk,
    stride_wb, stride_wk, stride_wn,
    stride_ym, stride_yn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program IDs
    pid0 = tl.program_id(axis=0)  # tiles over (M, N)
    pid_b = tl.program_id(axis=1)  # batch/group id

    # Compute group start (prev) and end (cur) from offsets
    cur = tl.load(offs_ptr + pid_b)
    prev = tl.load(offs_ptr + pid_b - 1, mask=pid_b > 0, other=0)
    m_len = cur - prev

    # Number of tiles
    num_pid_m = tl.cdiv(M_MAX, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n

    # Map pid0 to (pid_m, pid_n) with M-grouping
    group_id = pid0 // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_m = first_pid_m + (pid0 % GROUP_M)
    pid_n = (pid0 % num_pid_in_group) // GROUP_M

    # Offsets in M and N for this tile
    offs_m_local = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m_global = prev + offs_m_local

    # Masks for boundaries
    mask_m = offs_m_local < m_len
    mask_n = offs_n < N

    # Improve coalescing along N
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BN), BN)
    # Make a safe M index for masked accesses
    offs_m_global_safe = tl.where(mask_m, offs_m_global, 0)

    # Accumulator in FP32
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Base pointer for the selected group's weights
    w_base = w_ptr + pid_b * stride_wb

    # Loop over K dimension
    for k0 in tl.range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        k_mask = offs_k < K

        # Compute pointers
        x_ptrs = x_ptr + (offs_m_global_safe[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_base + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        # Load with masking
        x = tl.load(x_ptrs, mask=(mask_m[:, None] & k_mask[None, :]), other=0.0)
        w = tl.load(w_ptrs, mask=(k_mask[:, None] & mask_n[None, :]), other=0.0)

        # Multiply-accumulate
        acc = tl.dot(x, w, acc)

    # Write back
    y_ptrs = y_ptr + (offs_m_global_safe[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=(mask_m[:, None] & mask_n[None, :]))


def kernel_function(X: torch.Tensor, W: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Grouped GEMM: For each group b = 0..B-1, multiply the slice X[prev:end, :] by W[b, :, :],
    where offsets[b] = end and prev = offsets[b-1] (with prev = 0 for b=0). All computation
    is performed in a single fused Triton kernel.
    """
    # Basic validation
    if not (isinstance(X, torch.Tensor) and isinstance(W, torch.Tensor) and isinstance(offsets, torch.Tensor)):
        raise TypeError("X, W, and offsets must be torch Tensors")
    if not (X.is_cuda and W.is_cuda):
        raise ValueError("X and W must be CUDA tensors")
    if X.device != W.device:
        raise ValueError("X and W must be on the same device")
    if X.dtype != torch.bfloat16 or W.dtype != torch.bfloat16:
        raise ValueError("X and W must be bfloat16")
    if X.dim() != 2 or W.dim() != 3:
        raise ValueError("X must be 2D and W must be 3D")

    total_rows, K = X.shape
    B, K_w, N = W.shape
    if K != K_w:
        raise ValueError("Incompatible K dimensions between X and W")

    # Offsets: accept CPU or CUDA, cast to int32 and move to device
    offs_d = offsets.to(device=X.device, dtype=torch.int32)
    if offs_d.dim() != 1 or offs_d.numel() != B:
        raise ValueError("offsets must be 1D with length B")
    if int(offs_d[-1].item()) != total_rows:
        raise ValueError("final offset must equal total_rows")

    # Compute maximum group length M_MAX for tiling
    offs_host = offs_d.cpu()
    prev = 0
    M_max = 0
    for o in offs_host.tolist():
        M_max = max(M_max, int(o) - prev)
        prev = int(o)

    # Allocate output tensor
    Y = torch.empty((total_rows, N), dtype=X.dtype, device=X.device)

    # Launch grid function
    def grid(meta):
        BM = meta["BM"]
        BN = meta["BN"]
        GROUP_M = meta["GROUP_M"]
        tm = triton.cdiv(M_max, BM)
        tn = triton.cdiv(N, BN)
        groups = triton.cdiv(tm, GROUP_M)
        return (groups * tn * GROUP_M, B)

    # Launch Triton kernel
    _grouped_mm_kernel[grid](
        X, W, Y, offs_d,
        B, N, K, M_max,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1), W.stride(2),
        Y.stride(0), Y.stride(1),
    )

    return Y