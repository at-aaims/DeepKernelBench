import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _grouped_gemm_kernel(
    x_ptr,                   # *bf16  [sum(M_g), K]
    w_ptr,                   # *bf16  [B, K, N]
    y_ptr,                   # *bf16  [sum(M_g), N]
    offs_ptr,                # *i32   [B] cumulative sums
    idx_ptr,                 # *i32   [num_groups_this_launch] indices of groups handled by this launch
    M, N, K,                 # int32  (M here is per-launch max M over picked groups)
    stride_xm, stride_xk,    # strides for X
    stride_wb, stride_wk, stride_wn,  # strides for W
    stride_ym, stride_yn,    # strides for Y
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    bid   = tl.program_id(2)

    # Which logical group (batch id) this program handles
    gid = tl.load(idx_ptr + bid, mask=True, other=0)

    # Compute start/end rows for this group using cumulative offsets
    end_g   = tl.load(offs_ptr + gid, mask=True, other=0)
    start_g = tl.load(offs_ptr + gid - 1, mask=gid > 0, other=0)
    glen    = end_g - start_g  # rows in this group

    tile_m = pid_m * BLOCK_M
    tile_n = pid_n * BLOCK_N

    # Early exit if this program's tile is completely out-of-bounds for this group
    if (tile_m >= glen) or (tile_n >= N):
        return

    # Base pointers for this group's slices
    x_g_ptr = x_ptr + start_g * stride_xm
    y_g_ptr = y_ptr + start_g * stride_ym
    w_g_ptr = w_ptr + gid * stride_wb

    # Offsets for this tile
    offs_m = tile_m + tl.arange(0, BLOCK_M)
    offs_n = tile_n + tl.arange(0, BLOCK_N)
    # Masks for bounds
    m_mask = offs_m < glen
    n_mask = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Compute pointers for A (X group slice) and B (W[gid])
        a_ptrs = x_g_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        b_ptrs = w_g_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc)

    # Store results
    c_ptrs = y_g_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def kernel_function(X: torch.Tensor, W: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Grouped GEMM using Triton. Computes out = torch._grouped_mm(X, W, offs=offsets)
    where:
      - X is [sum(M_g), K] bf16
      - W is [B, K, N] bf16
      - offsets is cumulative row counts [B] int32, with offsets[b] = sum_{i<=b} M_i
    """
    # Validate inputs
    if not (isinstance(X, torch.Tensor) and isinstance(W, torch.Tensor) and isinstance(offsets, torch.Tensor)):
        raise TypeError("X, W, and offsets must be torch.Tensors")
    if X.device.type != "cuda" or W.device.type != "cuda":
        raise RuntimeError("X and W must be CUDA tensors")
    if X.dtype != torch.bfloat16 or W.dtype != torch.bfloat16:
        raise TypeError("X and W must be bfloat16")
    if X.ndim != 2 or W.ndim != 3:
        raise ValueError("Expected X [M_total, K] and W [B, K, N]")

    device = X.device
    if offsets.device != device:
        offsets = offsets.to(device=device)
    if offsets.dtype != torch.int32:
        offsets = offsets.to(dtype=torch.int32, device=device)

    B, K, N = W.shape
    if offsets.numel() != B:
        raise ValueError("offsets length must equal B")
    if X.shape[1] != K:
        raise ValueError("X.shape[1] must equal W.shape[1] (K)")

    # Validate offsets and compute per-group lengths on CPU
    offs_list = offsets.detach().cpu().tolist()
    if len(offs_list) == 0:
        raise ValueError("offsets must have at least one element")

    M_total = int(offs_list[-1])
    if X.shape[0] != M_total:
        raise ValueError("X rows must equal last element of offsets (total rows)")

    prev = 0
    lens = []
    for e in offs_list:
        glen = int(e) - prev
        if glen < 0:
            raise ValueError("offsets must be a non-decreasing cumulative sum")
        lens.append(glen)
        prev = int(e)

    # Allocate output
    Y = torch.empty((M_total, N), dtype=torch.bfloat16, device=device)

    # Strides
    stride_xm, stride_xk = X.stride(0), X.stride(1)
    stride_wb, stride_wk, stride_wn = W.stride(0), W.stride(1), W.stride(2)
    stride_ym, stride_yn = Y.stride(0), Y.stride(1)

    # Bucket groups by their lengths to pick a good BLOCK_M
    buckets = {32: [], 64: [], 128: []}
    for gid, glen in enumerate(lens):
        if glen <= 32:
            buckets[32].append(gid)
        elif glen <= 64:
            buckets[64].append(gid)
        else:
            buckets[128].append(gid)

    # Launch one kernel per bucket
    for BM in [32, 64, 128]:
        gids = buckets[BM]
        if len(gids) == 0:
            continue
        idx = torch.tensor(gids, dtype=torch.int32, device=device)
        m_max = max(lens[g] for g in gids)

        def grid(meta):
            BN = meta["BLOCK_N"]
            return (triton.cdiv(N, BN), triton.cdiv(m_max, BM), len(gids))

        _grouped_gemm_kernel[grid](
            X, W, Y, offsets, idx,
            m_max, N, K,
            stride_xm, stride_xk,
            stride_wb, stride_wk, stride_wn,
            stride_ym, stride_yn,
        )

    return Y