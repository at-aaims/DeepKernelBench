import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def _grouped_gemm_kernel(
    x_ptr,          # *bfloat16, [M_total, K]
    w_ptr,          # *bfloat16, [B, K, N]
    y_ptr,          # *bfloat16, [M_total, N]
    task_rows_ptr,  # *int32, [T]
    task_gids_ptr,  # *int32, [T]
    task_lens_ptr,  # *int32, [T]
    M_total, N, K,  # ints
    stride_xm, stride_xk,
    stride_wb, stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)  # column tile id
    pid_t = tl.program_id(1)  # task id in [0, T)

    # per-task metadata
    row_start = tl.load(task_rows_ptr + pid_t)
    gid = tl.load(task_gids_ptr + pid_t)
    m_len = tl.load(task_lens_ptr + pid_t)

    # N-tile
    n_start = pid_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # base pointer for this group's weight
    w_base = w_ptr + gid * stride_wb

    # block pointers
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M_total, K),
        strides=(stride_xm, stride_xk),
        offsets=(row_start, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_block_ptr = tl.make_block_ptr(
        base=w_base,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, n_start),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # iterate over K dimension
    k_loops = tl.cdiv(K, BLOCK_K)
    for _ in tl.range(0, k_loops):
        # Load tiles with boundary protection on both K and N/M as needed.
        # For X: may go out-of-bounds on M (tail tiles) and on K
        x_tile = tl.load(
            x_block_ptr,
            boundary_check=(0, 1),
            padding_option='zero',
            cache_modifier='.ca',
            eviction_policy='evict_last'
        )
        # For W: may go out-of-bounds on N (tail of N tile) and on K
        w_tile = tl.load(
            w_block_ptr,
            boundary_check=(0, 1),
            padding_option='zero',
            cache_modifier='.ca',
            eviction_policy='evict_last'
        )
        acc = tl.dot(x_tile, w_tile, acc)
        # Advance block pointers (tl.advance returns a new pointer; must assign it)
        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_K, 0))

    # Store result with masks for tail rows and columns
    offs_m_rel = tl.arange(0, BLOCK_M)
    offs_m = row_start + offs_m_rel
    mask_m = offs_m_rel < m_len

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(X: torch.Tensor, W: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Grouped GEMM implemented in Triton, equivalent to:
      out = torch._grouped_mm(X, W, offs=offsets)

    Args:
      X: [sum(group_lens), K], bfloat16, CUDA
      W: [B, K, N], bfloat16, CUDA
      offsets: [B], int32/int64, cumulative row counts per group (cumsum of group_lens)

    Returns:
      Y: [sum(group_lens), N], bfloat16, CUDA
    """
    # Basic validation and setup (no math here; all compute in Triton kernels)
    if not (isinstance(X, torch.Tensor) and isinstance(W, torch.Tensor) and isinstance(offsets, torch.Tensor)):
        raise TypeError("X, W, and offsets must be torch.Tensors")

    if not (X.is_cuda and W.is_cuda):
        raise RuntimeError("X and W must be CUDA tensors")

    device = X.device
    if W.device != device:
        raise ValueError("X and W must be on the same device")

    if offsets.device != device:
        offsets = offsets.to(device)

    if X.dtype != torch.bfloat16 or W.dtype != torch.bfloat16:
        raise TypeError("X and W must be bfloat16")

    if X.ndim != 2 or W.ndim != 3:
        raise ValueError("Expected X [M_total, K] and W [B, K, N]")

    if offsets.ndim != 1:
        raise ValueError("offsets must be 1D")

    if offsets.dtype not in (torch.int32, torch.int64):
        offsets = offsets.to(dtype=torch.int32)
    if offsets.dtype != torch.int32:
        offsets = offsets.int()

    B = W.shape[0]
    K = W.shape[1]
    N = W.shape[2]
    if offsets.numel() != B:
        raise ValueError("offsets length must equal B")
    if X.shape[1] != K:
        raise ValueError("X.shape[1] must equal W.shape[1] (K)")

    # Compute starts and verify M_total from offsets
    offs_list = offsets.detach().cpu().tolist()
    if len(offs_list) == 0:
        raise ValueError("offsets must have at least one element")
    M_total = int(offs_list[-1])
    if X.shape[0] != M_total:
        raise ValueError("X rows must equal last offset")

    starts = []
    prev = 0
    for e in offs_list:
        starts.append(prev)
        prev = int(e)

    # Allocate output
    Y = torch.empty((M_total, N), dtype=torch.bfloat16, device=device)

    # Choose tiling sizes; auto-tuner will explore configs.
    # We still need a base BLOCK_M for task partitioning; use 128 to keep tasks coarse.
    BASE_BLOCK_M = 128

    # Build task lists (rows, group ids, and lengths). We partition each group
    # into tiles of size BASE_BLOCK_M, with a possible tail < BASE_BLOCK_M.
    task_rows = []
    task_gids = []
    task_lens = []
    for g in range(B):
        g_start = starts[g]
        g_end = offs_list[g]
        glen = g_end - g_start
        if glen <= 0:
            continue
        full_blocks = glen // BASE_BLOCK_M
        rem = glen - full_blocks * BASE_BLOCK_M
        # Full tiles
        for t in range(full_blocks):
            task_rows.append(g_start + t * BASE_BLOCK_M)
            task_gids.append(g)
            task_lens.append(BASE_BLOCK_M)
        # Tail tile
        if rem > 0:
            task_rows.append(g_start + full_blocks * BASE_BLOCK_M)
            task_gids.append(g)
            task_lens.append(rem)

    if len(task_rows) == 0:
        # Degenerate case: nothing to do
        return Y

    task_rows_t = torch.tensor(task_rows, dtype=torch.int32, device=device)
    task_gids_t = torch.tensor(task_gids, dtype=torch.int32, device=device)
    task_lens_t = torch.tensor(task_lens, dtype=torch.int32, device=device)
    T = task_rows_t.numel()

    # Strides
    stride_xm, stride_xk = X.stride(0), X.stride(1)
    stride_wb, stride_wk, stride_wn = W.stride(0), W.stride(1), W.stride(2)
    stride_ym, stride_yn = Y.stride(0), Y.stride(1)

    # Provide grid as a lambda depending on autotuned BLOCK_N
    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_N']), T)

    # Launch single fused kernel handling both full and tail tiles via mask
    _grouped_gemm_kernel[grid](
        X, W, Y,
        task_rows_t, task_gids_t, task_lens_t,
        M_total, N, K,
        stride_xm, stride_xk,
        stride_wb, stride_wk, stride_wn,
        stride_ym, stride_yn,
    )

    return Y

# Note on fusion choice:
# We consolidated the previous "full" and "tail" kernels into a single fused kernel
# that accepts per-task lengths and masks both loads and stores. This avoids launching
# two separate kernels and ensures all grouped-GEMM work is performed in one pass,
# maximizing fusion and minimizing overhead while still allowing autotuning.