import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _grouped_gemm_tc_kernel(
    x_ptr,            # *bf16 [M_total, K]
    w_ptr,            # *bf16 [B, K, N]
    y_ptr,            # *bf16 [M_total, N]
    offs_ptr,         # *i32  [B] cumulative group offsets
    M, N, K,          # i32   total M (for autotune key), N, K
    stride_xm, stride_xk,
    stride_wb, stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    gid   = tl.program_id(axis=2)

    end_g = tl.load(offs_ptr + gid)
    start_g = tl.load(offs_ptr + gid - 1, mask=gid > 0, other=0)
    gM = end_g - start_g

    x_base = x_ptr + start_g * stride_xm
    y_base = y_ptr + start_g * stride_ym
    w_group_ptr = w_ptr + gid * stride_wb

    n0 = pid_n * BLOCK_N
    m0 = pid_m * BLOCK_M
    tl.multiple_of(n0, 64)

    if m0 >= gM:
        return

    rm = tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_shape_m = gM
    a_shape_k = K
    b_shape_k = K
    b_shape_n = N

    a_block_ptr = tl.make_block_ptr(
        base=x_base,
        shape=(a_shape_m, a_shape_k),
        strides=(stride_xm, stride_xk),
        offsets=(m0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=w_group_ptr,
        shape=(b_shape_k, b_shape_n),
        strides=(stride_wk, stride_wn),
        offsets=(0, n0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    k_tiles = tl.cdiv(K, BLOCK_K)
    for _ in range(0, k_tiles):
        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero", cache_modifier=".cg")
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_ptrs = y_base + (m0 + rm)[:, None] * stride_ym + (n0 + rn)[None, :] * stride_yn
    mask_m = (m0 + rm) < gM
    mask_n = (n0 + rn) < N
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(X: torch.Tensor, W: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if not (isinstance(X, torch.Tensor) and isinstance(W, torch.Tensor) and isinstance(offsets, torch.Tensor)):
        raise TypeError("X, W, and offsets must be torch.Tensors")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for this kernel.")

    device = X.device if X.is_cuda else (W.device if W.is_cuda else torch.device("cuda"))
    if not X.is_cuda:
        X = X.to(device=device)
    if not W.is_cuda:
        W = W.to(device=device)
    if not offsets.is_cuda:
        offsets = offsets.to(device=device)

    if X.ndim != 2 or W.ndim != 3:
        raise ValueError("Expected X [M_total, K] and W [B, K, N]")
    if X.dtype != torch.bfloat16 or W.dtype != torch.bfloat16:
        raise TypeError("X and W must be bfloat16 tensors")

    B = int(W.shape[0])
    K = int(W.shape[1])
    N = int(W.shape[2])

    if offsets.ndim != 1 or int(offsets.numel()) != B:
        raise ValueError("offsets must have shape [B]")
    if int(X.shape[1]) != K:
        raise ValueError("K mismatch between X and W")

    if offsets.dtype != torch.int32:
        offsets = offsets.to(dtype=torch.int32)

    offs_list = offsets.detach().cpu().tolist()
    if len(offs_list) == 0:
        raise ValueError("offsets must have at least one element")
    M_total = int(offs_list[-1])
    if int(X.shape[0]) != M_total:
        raise ValueError("X rows must equal offsets[-1] (M_total)")

    prev = 0
    m_max = 0
    for e in offs_list:
        glen = int(e) - prev
        if glen < 0:
            raise ValueError("offsets must be non-decreasing")
        if glen > m_max:
            m_max = glen
        prev = int(e)

    if W.stride(2) != 1:
        W = W.contiguous()

    Y = torch.empty((M_total, N), dtype=torch.bfloat16, device=device)

    stride_xm, stride_xk = X.stride(0), X.stride(1)
    stride_wb, stride_wk, stride_wn = W.stride(0), W.stride(1), W.stride(2)
    stride_ym, stride_yn = Y.stride(0), Y.stride(1)

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        return (triton.cdiv(N, BN), triton.cdiv(m_max, BM), B)

    _grouped_gemm_tc_kernel[grid](
        X, W, Y, offsets,
        M_total, N, K,
        stride_xm, stride_xk,
        stride_wb, stride_wk, stride_wn,
        stride_ym, stride_yn,
    )

    return Y