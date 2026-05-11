import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 8,
                "BLOCK_N": 64,
                "BLOCK_D": 64,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 64,
                "BLOCK_D": 64,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 128,
                "BLOCK_D": 64,
            },
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["S", "HEAD_DIM", "IS_BF16"],
)
@triton.jit
def _sdpa_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    scale,  # sm_scale = 1/sqrt(HEAD_DIM)
    B, H, S, D,  # dimensions
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Fused scaled dot-product attention (SDPA) forward pass (no dropout, non-causal)
    Computation per head (for a block of queries):
      - For each query block (M):
          - Stream over K/V blocks (N) with online softmax:
            qk = Q[M, D] @ K[N, D]^T
            p  = softmax(qk) combined across N tiles using online normalization
            acc = acc * alpha + p @ V[N, D]
          - Normalize acc by l_i at the end and store to output.
    The kernel processes:
      pid_m: query block along sequence length
      pid_bh: flattened (batch, head) index
    """
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    # Derive batch/head index
    h_idx = pid_bh % H
    b_idx = pid_bh // H

    # Base pointers for this (b, h)
    q_head_ptr = q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_head_ptr = k_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_head_ptr = v_ptr + b_idx * stride_vb + h_idx * stride_vh
    o_head_ptr = o_ptr + b_idx * stride_ob + h_idx * stride_oh

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_small = tl.arange(0, BLOCK_D)
    offs_d_all = tl.arange(0, HEAD_DIM)

    # Masks for bounds
    m_mask = offs_m < S

    # Online softmax state per query row
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Accumulator for output [BLOCK_M, HEAD_DIM] in fp32
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Loop over key/value blocks along sequence dimension (N dimension)
    for start_n in tl.range(0, S, BLOCK_N):
        # qk tile accumulator in fp32
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Iterate over D dimension in BLOCK_D chunks to compute qk
        for d0 in range(0, HEAD_DIM, BLOCK_D):
            cd = d0 + offs_d_small  # current D chunk offsets
            # Load Q sub-tile: [BLOCK_M, BLOCK_D]
            q_ptrs = q_head_ptr + offs_m[:, None] * stride_qs + cd[None, :] * stride_qd
            q_sub = tl.load(q_ptrs, mask=(m_mask[:, None] & (cd[None, :] < HEAD_DIM)), other=0.0)

            # Load K sub-tile as [BLOCK_D, BLOCK_N] for matmul
            k_ptrs = k_head_ptr + (start_n + offs_n[None, :]) * stride_ks + cd[:, None] * stride_kd
            n_mask = (start_n + offs_n) < S
            k_sub = tl.load(k_ptrs, mask=((cd[:, None] < HEAD_DIM) & n_mask[None, :]), other=0.0)

            # Accumulate qk += Q_sub @ K_sub
            qk += tl.dot(q_sub, k_sub)

        # Apply scaling factor
        qk = qk * scale

        # Online softmax update for numerical stability
        row_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, row_max)
        # p = exp(qk - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_new)

        # Update accumulator with rescaling
        acc = acc * alpha[:, None]

        # Load V tile [BLOCK_N, HEAD_DIM]
        v_ptrs = v_head_ptr + (start_n + offs_n)[:, None] * stride_vs + offs_d_all[None, :] * stride_vd
        n_mask = (start_n + offs_n) < S
        v_tile = tl.load(
            v_ptrs,
            mask=(n_mask[:, None] & (offs_d_all[None, :] < HEAD_DIM)),
            other=0.0
        )

        # Cast p to match V dtype before dot
        if IS_BF16:
            p_cast = p.to(tl.bfloat16)
        else:
            p_cast = p.to(tl.float16)

        # acc += p @ V_tile
        acc = tl.dot(p_cast, v_tile, acc)

        # Commit softmax running stats
        l_i = l_new
        m_i = m_new

    # Final normalization: divide by l_i
    acc = acc / l_i[:, None]

    # Store result to output in the output pointer dtype
    out_ptrs = o_head_ptr + offs_m[:, None] * stride_os + offs_d_all[None, :] * stride_od
    out_mask = (offs_m[:, None] < S) & (offs_d_all[None, :] < HEAD_DIM)
    out_vals = acc.to(o_ptr.dtype.element_ty)
    tl.store(out_ptrs, out_vals, mask=out_mask)


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Fused scaled dot-product attention (SDPA) forward pass implemented with a single Triton kernel.

    What is fused:
    - QK^T matmul (streaming over K/V blocks)
    - Online softmax normalization (per query row) for numerical stability
    - Multiplication by V and accumulation into the output
    - Final normalization by the softmax partition function
    All stages are fused into one pass over K/V tiles to avoid forming the SxS attention matrix,
    minimizing memory traffic and kernel launch overhead.

    Wrapper responsibilities only:
    - Validate shapes/dtypes/devices
    - Allocate output tensor and compute scaling factor
    - Configure grid and launch the Triton kernel
    No PyTorch compute ops (matmul, softmax, etc.) are used here.

    Expected input shapes: [B, H, S, D]
    - No dropout, non-causal
    - Dtype: bfloat16 preferred; float16 supported as fallback
    """
    # Basic validations
    assert Q.device.type == "cuda" and K.device.type == "cuda" and V.device.type == "cuda", "All tensors must be on CUDA."
    assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape [B, H, S, D]."
    assert Q.dtype in (torch.bfloat16, torch.float16), "Supported dtypes: bfloat16 or float16."
    assert K.dtype == Q.dtype and V.dtype == Q.dtype, "Q, K, V dtypes must match."

    B, H, S, D = Q.shape

    # Scaling for SDPA: 1/sqrt(D)
    sm_scale = 1.0 / math.sqrt(D)

    # Allocate output, same dtype as inputs
    O = torch.empty_like(Q)

    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
    stride_ob, stride_oh, stride_os, stride_od = O.stride()

    # Grid: one program per (query block, head)
    # Choose launch parameters via autotune; still need grid sizes here.
    # We use BLOCK_M from the autotuned META. Triton will substitute during launch.
    def grid(META):
        BM = META["BLOCK_M"]
        return (triton.cdiv(S, BM), B * H)

    _sdpa_kernel[grid](
        Q, K, V, O,
        sm_scale,
        B, H, S, D,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        IS_BF16=int(Q.dtype == torch.bfloat16),
        HEAD_DIM=D,
    )
    return O