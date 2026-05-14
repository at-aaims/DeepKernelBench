import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "BLOCK_D_OUT": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "BLOCK_D_OUT": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "BLOCK_D_OUT": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64, "BLOCK_D_OUT": 64}, num_warps=4, num_stages=4),
    ],
    key=["S", "D", "IS_BF16"],
)
@triton.jit
def _sdpa_fwd_kernel(
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
    BLOCK_K: tl.constexpr,
    BLOCK_D_OUT: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    b_idx = pid_bh // H
    h_idx = pid_bh % H

    q_head_ptr = q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_head_ptr = k_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_head_ptr = v_ptr + b_idx * stride_vb + h_idx * stride_vh
    o_head_ptr = o_ptr + b_idx * stride_ob + h_idx * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D_OUT)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in tl.range(0, S, BLOCK_N):
        n_idx = start_n + offs_n
        n_mask = n_idx < S

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for start_k in tl.range(0, D, BLOCK_K):
            k_idx = start_k + offs_k
            k_mask_vec = k_idx < D

            q_ptrs = q_head_ptr + (offs_m[:, None] * stride_qs + k_idx[None, :] * stride_qd)
            q_chunk = tl.load(
                q_ptrs,
                mask=(m_mask[:, None] & k_mask_vec[None, :]),
                other=0.0,
                eviction_policy="evict_last",
                cache_modifier=".ca",
            )

            k_ptrs = k_head_ptr + (n_idx[None, :] * stride_ks + k_idx[:, None] * stride_kd)
            k_chunk = tl.load(
                k_ptrs,
                mask=(k_mask_vec[:, None] & n_mask[None, :]),
                other=0.0,
                eviction_policy="evict_last",
                cache_modifier=".ca",
            )

            qk = tl.dot(q_chunk, k_chunk, qk)

        qk = qk * scale
        qk = tl.where(n_mask[None, :], qk, -float("inf"))

        row_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, row_max)

        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        w = p / l_new[:, None]
        s_factor = (l_i * alpha) / l_new

        w_cast = w.to(tl.bfloat16) if IS_BF16 else w.to(tl.float16)

        for start_d in tl.range(0, D, BLOCK_D_OUT):
            d_idx = start_d + offs_d
            d_mask = d_idx < D

            v_ptrs = v_head_ptr + (n_idx[:, None] * stride_vs + d_idx[None, :] * stride_vd)
            v_block = tl.load(
                v_ptrs,
                mask=(n_mask[:, None] & d_mask[None, :]),
                other=0.0,
                eviction_policy="evict_last",
                cache_modifier=".ca",
            )

            acc_sub = tl.zeros([BLOCK_M, BLOCK_D_OUT], dtype=tl.float32)
            acc_sub = tl.dot(w_cast, v_block, acc_sub)

            need_prev = l_i > 0
            prev_load_mask = (m_mask[:, None] & need_prev[:, None] & d_mask[None, :])

            o_ptrs = o_head_ptr + (offs_m[:, None] * stride_os + d_idx[None, :] * stride_od)
            o_prev = tl.load(
                o_ptrs,
                mask=prev_load_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)

            y_new = o_prev * s_factor[:, None] + acc_sub

            tl.store(
                o_ptrs,
                y_new.to(o_ptr.dtype.element_ty),
                mask=(m_mask[:, None] & d_mask[None, :]),
                eviction_policy="evict_last",
            )

        l_i = l_new
        m_i = m_new


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert Q.device.type == "cuda" and K.device.type == "cuda" and V.device.type == "cuda"
    assert Q.shape == K.shape == V.shape
    assert Q.dtype in (torch.bfloat16, torch.float16)
    assert K.dtype == Q.dtype and V.dtype == Q.dtype

    B, H, S, D = Q.shape

    O = torch.empty_like(Q)

    stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
    stride_ob, stride_oh, stride_os, stride_od = O.stride()

    def grid(META):
        return (triton.cdiv(S, META["BLOCK_M"]), B * H)

    sm_scale = 1.0 / math.sqrt(D)

    _sdpa_fwd_kernel[grid](
        Q, K, V, O,
        scale=sm_scale,
        B=B, H=H, S=S, D=D,
        stride_qb=stride_qb, stride_qh=stride_qh, stride_qs=stride_qs, stride_qd=stride_qd,
        stride_kb=stride_kb, stride_kh=stride_kh, stride_ks=stride_ks, stride_kd=stride_kd,
        stride_vb=stride_vb, stride_vh=stride_vh, stride_vs=stride_vs, stride_vd=stride_vd,
        stride_ob=stride_ob, stride_oh=stride_oh, stride_os=stride_os, stride_od=stride_od,
        IS_BF16=int(Q.dtype == torch.bfloat16),
    )
    return O