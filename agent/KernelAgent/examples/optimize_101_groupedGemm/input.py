import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _grouped_gemm_kernel(
    x_ptr,          # *bfloat16, shape [M_total, K]
    w_ptr,          # *bfloat16, shape [B, K, N]
    y_ptr,          # *bfloat16, shape [M_total, N]
    offs_ptr,       # *int32, shape [B] cumulative row offsets (exclusive end)
    M, N, K,        # ints: M is max group length, N, K are dimensions
    stride_xm, stride_xk,
    stride_wb, stride_wk, stride_wn,
    stride_ym, stride_yn,
    # compile-time tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Triton kernel performing grouped GEMM:
      For each group g, rows in [start_g, end_g) of X multiply W[g] to produce Y rows in the same slice.
    The launch grid is:
      - pid_n over columns (N) in tiles of BLOCK_N
      - pid_m over rows within group (max over all groups) in tiles of BLOCK_M
      - pid_g over groups (B)

    Memory math is done entirely in Triton. Accumulation is in fp32 for numerical stability and cast to bf16 on store.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    gid = tl.program_id(2)  # group id

    # Load start/end row indices for this group from offsets
    end_g = tl.load(offs_ptr + gid, eviction_policy="evict_last")
    # start_g needs a uniform branch to avoid OOB read when gid == 0
    start_g = tl.zeros((), dtype=tl.int32)
    if gid == 0:
        start_g = tl.zeros((), dtype=tl.int32)
    else:
        start_g = tl.load(offs_ptr + (gid - 1), eviction_policy="evict_last")

    # Compute row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = start_g + offs_m  # global row indices in X/Y for this group
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for boundaries
    mask_m = rows < end_g
    mask_n = offs_n < N

    # Make indices more friendly for codegen
    rows = tl.where(mask_m, rows, 0)
    offs_n = tl.where(mask_n, offs_n, 0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    offs_k = tl.arange(0, BLOCK_K)
    for k0 in tl.range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        mask_k = k_idx < K

        # Pointers to X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (rows[:, None] * stride_xm + k_idx[None, :] * stride_xk)
        # Load X with proper masking
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Pointers to W tile for group gid: [BLOCK_K, BLOCK_N]
        w_base = w_ptr + gid * stride_wb
        w_ptrs = w_base + (k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc = tl.dot(x_tile, w_tile, acc)

    # Store results to Y
    y_ptrs = y_ptr + (rows[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(X: torch.Tensor, W: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Grouped GEMM using a single fused Triton kernel.

    Fused stages:
    - Per-group selection of row slices from the large concatenated X using offsets (no separate slicing kernel).
    - Matrix multiplication X_g @ W[g] for each group g is computed directly in the same kernel for all groups.
    - Accumulation is done in fp32; cast to bf16 is applied as the epilogue in the same kernel.
    This design fuses group routing and matmul, avoiding intermediate allocations or extra kernel launches.

    Args:
        X: Tensor of shape [sum(M_g), K], dtype bfloat16, on CUDA.
           In the provided test, shape is [B*M, K] with B=4 and M=4096.
        W: Tensor of shape [B, K, N], dtype bfloat16, on CUDA.
        offsets: Tensor of shape [B], dtype int32 or int64, cumulative row ends per group (exclusive).

    Returns:
        Y: Tensor of shape [sum(M_g), N], dtype bfloat16, on the same device as X.
    """
    # Basic validation
    if not (isinstance(X, torch.Tensor) and isinstance(W, torch.Tensor) and isinstance(offsets, torch.Tensor)):
        raise TypeError("X, W, and offsets must be torch.Tensors")
    if not X.is_cuda or not W.is_cuda or not offsets.is_cuda:
        # offsets can be provided on CPU in some scenarios; move to device for kernel
        # The operation is performed on the device of X/W
        device = X.device if X.is_cuda else (W.device if W.is_cuda else None)
        if device is None or device.type != "cuda":
            raise RuntimeError("CUDA device required for this kernel.")
        offsets = offsets.to(device=device, dtype=torch.int32)
    device = X.device
    if W.device != device:
        raise ValueError(f"Device mismatch: X on {X.device}, W on {W.device}")
    if offsets.device != device:
        offsets = offsets.to(device=device)

    if X.dtype != torch.bfloat16 or W.dtype != torch.bfloat16:
        raise TypeError(f"X and W must be bfloat16; got {X.dtype} and {W.dtype}")
    if X.ndim != 2 or W.ndim != 3:
        raise ValueError(f"Expected X [M_total, K] and W [B, K, N]; got {X.shape} and {W.shape}")
    if offsets.ndim != 1:
        raise ValueError(f"Expected offsets [B]; got {offsets.shape}")

    B = W.shape[0]
    K = W.shape[1]
    N = W.shape[2]
    if offsets.numel() != B:
        raise ValueError(f"offsets length {offsets.numel()} must equal B={B}")
    if X.shape[1] != K:
        raise ValueError(f"Incompatible K: X.shape[1]={X.shape[1]} vs W.shape[1]={K}")
    if offsets.dtype not in (torch.int32, torch.int64):
        offsets = offsets.to(dtype=torch.int32)
    if offsets.dtype != torch.int32:
        offsets = offsets.int()

    # Fetch cumulative ends to CPU to compute group max length for grid scheduling.
    # This is a small integer-only operation used strictly for launch configuration.
    offs_list = offsets.detach().cpu().tolist()
    if len(offs_list) == 0:
        raise ValueError("offsets must have at least one element")
    M_total = int(offs_list[-1])
    if X.shape[0] != M_total:
        raise ValueError(f"X rows {X.shape[0]} must equal last offset {M_total}")
    # Compute per-group lengths to obtain max for grid height
    prev = 0
    m_max = 0
    for e in offs_list:
        glen = int(e) - prev
        if glen < 0:
            raise ValueError("offsets must be non-decreasing cumulative ends")
        if glen > m_max:
            m_max = glen
        prev = int(e)
    if m_max == 0:
        # Degenerate case: all groups empty
        return torch.zeros((M_total, N), dtype=X.dtype, device=device)

    # Allocate output
    Y = torch.empty((M_total, N), dtype=torch.bfloat16, device=device)

    # Strides (in elements, not bytes)
    stride_xm, stride_xk = X.stride(0), X.stride(1)
    stride_wb, stride_wk, stride_wn = W.stride(0), W.stride(1), W.stride(2)
    stride_ym, stride_yn = Y.stride(0), Y.stride(1)

    # Tile sizes; autotuner will pick the best among provided configs, but we need them for grid function
    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (
            triton.cdiv(N, BN),         # pid_n
            triton.cdiv(m_max, BM),     # pid_m (per group)
            B,                          # pid_g
        )

    _grouped_gemm_kernel[grid](
        X, W, Y, offsets,
        m_max, N, K,
        stride_xm, stride_xk,
        stride_wb, stride_wk, stride_wn,
        stride_ym, stride_yn,
    )
    return Y