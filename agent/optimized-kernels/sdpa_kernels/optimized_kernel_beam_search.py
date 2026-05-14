import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_D": 64},  num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 128}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512, "BLOCK_D": 64},  num_warps=16, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_D": 128}, num_warps=4,  num_stages=4),
    ],
    key=["S", "HEAD_DIM", "IS_BF16"],
)
@triton.jit
def _sdpa_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    scale,
    B, H, S, D,
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
    # Program IDs
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    h_idx = pid_bh % H
    b_idx = pid_bh // H

    # Base pointers per (batch, head)
    q_head_ptr = q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_head_ptr = k_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_head_ptr = v_ptr + b_idx * stride_vb + h_idx * stride_vh
    o_head_ptr = o_ptr + b_idx * stride_ob + h_idx * stride_oh

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    m_mask = offs_m < S

    # Online softmax accumulators per row
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Iterate over K/V blocks along the sequence dimension
    for start_n in tl.range(0, S, BLOCK_N):
        n_mask = (start_n + offs_n) < S

        # Compute QK^T block
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for d0 in range(0, HEAD_DIM, BLOCK_D):
            cd = d0 + offs_d

            # Load Q sub-block: [BLOCK_M, BLOCK_D]
            q_ptrs = q_head_ptr + offs_m[:, None] * stride_qs + cd[None, :] * stride_qd
            q_sub = tl.load(
                q_ptrs,
                mask=(m_mask[:, None] & (cd[None, :] < HEAD_DIM)),
                other=0.0,
                cache_modifier=".ca",
                eviction_policy="evict_last",
            )

            # Load K sub-block (transposed layout for dot): [BLOCK_D, BLOCK_N]
            k_ptrs = k_head_ptr + (start_n + offs_n[None, :]) * stride_ks + cd[:, None] * stride_kd
            k_sub = tl.load(
                k_ptrs,
                mask=((cd[:, None] < HEAD_DIM) & n_mask[None, :]),
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            qk += tl.dot(q_sub, k_sub)

        # Scale scores
        qk = qk * scale

        # Online softmax update
        row_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, row_max)
        p = tl.exp(qk - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_new)

        # Cast probabilities to match V dtype for faster dot on tensor cores
        p_cast = p.to(tl.bfloat16) if IS_BF16 else p.to(tl.float16)

        # Accumulate P @ V into output buffer (stored as running numerators)
        for d0 in range(0, HEAD_DIM, BLOCK_D):
            cd = d0 + offs_d
            acc_mask = (m_mask[:, None] & (cd[None, :] < HEAD_DIM))

            v_ptrs = v_head_ptr + (start_n + offs_n)[:, None] * stride_vs + cd[None, :] * stride_vd
            v_sub = tl.load(
                v_ptrs,
                mask=(n_mask[:, None] & (cd[None, :] < HEAD_DIM)),
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            )

            # out_sub is in fp32 by default for fp16/bf16 inputs
            out_sub = tl.dot(p_cast, v_sub)

            o_ptrs_chunk = o_head_ptr + offs_m[:, None] * stride_os + cd[None, :] * stride_od
            if start_n == 0:
                prev_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
            else:
                prev_vals = tl.load(o_ptrs_chunk, mask=acc_mask, other=0.0)
                prev_acc = prev_vals.to(tl.float32)

            updated = prev_acc * alpha[:, None] + out_sub
            tl.store(o_ptrs_chunk, updated.to(o_ptr.dtype.element_ty), mask=acc_mask)

        # Commit new m_i and l_i
        l_i = l_new
        m_i = m_new

    # Final normalization: divide running numerators by l_i
    for d0 in range(0, HEAD_DIM, BLOCK_D):
        cd = d0 + offs_d
        out_mask = (m_mask[:, None] & (cd[None, :] < HEAD_DIM))
        o_ptrs_chunk = o_head_ptr + offs_m[:, None] * stride_os + cd[None, :] * stride_od

        accum_vals = tl.load(o_ptrs_chunk, mask=out_mask, other=0.0).to(tl.float32)
        accum_vals = accum_vals / l_i[:, None]
        tl.store(o_ptrs_chunk, accum_vals.to(o_ptr.dtype.element_ty), mask=out_mask)


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # Runtime validation and allocation only; all math resides in Triton kernel.
    assert Q.device.type == "cuda" and K.device.type == "cuda" and V.device.type == "cuda"
    assert Q.shape == K.shape == V.shape
    assert Q.dtype in (torch.bfloat16, torch.float16)
    assert K.dtype == Q.dtype and V.dtype == Q.dtype

    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)

    stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
    stride_ob, stride_oh, stride_os, stride_od = O.stride()

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