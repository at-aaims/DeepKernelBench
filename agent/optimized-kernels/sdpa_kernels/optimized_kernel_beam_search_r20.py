import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_D": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 128, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_D": 64, "GROUP_M": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_D": 64, "GROUP_M": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_D": 64, "GROUP_M": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128, "GROUP_M": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_D": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 128, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_D": 64, "GROUP_M": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_D": 64, "GROUP_M": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_D": 64, "GROUP_M": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 128, "GROUP_M": 16}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_D": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
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
    GROUP_M: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    h_idx = pid_bh % H
    b_idx = pid_bh // H

    # group reordering along M
    group_id = pid_m // GROUP_M
    group_start = group_id * GROUP_M
    pid_m_in_group = pid_m % GROUP_M
    start_m = (group_start + pid_m_in_group) * BLOCK_M

    # base pointers for this (b, h)
    q_head_ptr = q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_head_ptr = k_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_head_ptr = v_ptr + b_idx * stride_vb + h_idx * stride_vh
    o_head_ptr = o_ptr + b_idx * stride_ob + h_idx * stride_oh

    # indices and masks
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S

    dtype_in = tl.bfloat16 if IS_BF16 else tl.float16

    # Online softmax state per row
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over key/value blocks
    for start_n in tl.range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S

        # Compute QK^T block
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, HEAD_DIM, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            # Q: [M, D]
            q_ptrs = q_head_ptr + (offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd)
            # K: [D, N] (load transposed tile)
            k_ptrs = k_head_ptr + (offs_n[None, :] * stride_ks + offs_d[:, None] * stride_kd)
            q_sub = tl.load(q_ptrs, mask=(m_mask[:, None] & (offs_d[None, :] < HEAD_DIM)), other=0.0)
            k_sub = tl.load(k_ptrs, mask=((offs_d[:, None] < HEAD_DIM) & n_mask[None, :]), other=0.0)
            q_sub = q_sub.to(dtype_in)
            k_sub = k_sub.to(dtype_in)
            qk += tl.dot(q_sub, k_sub)

        # Scale and mask
        qk = qk * scale
        qk = tl.where(m_mask[:, None] & n_mask[None, :], qk, -float("inf"))

        # Online softmax update
        row_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, row_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        # Accumulate P @ V into O with running correction
        for d0 in tl.range(0, HEAD_DIM, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            v_ptrs = v_head_ptr + (offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd)
            v_sub = tl.load(v_ptrs, mask=(n_mask[:, None] & (offs_d[None, :] < HEAD_DIM)), other=0.0).to(dtype_in)

            # Cast p to input dtype for tensor cores; accumulate to fp32
            p_sub = p.to(dtype_in)
            delta = tl.dot(p_sub, v_sub)

            out_ptrs = o_head_ptr + (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
            out_mask = (m_mask[:, None] & (offs_d[None, :] < HEAD_DIM))
            out_old = tl.load(out_ptrs, mask=out_mask, other=0.0).to(tl.float32)
            out_new = out_old * alpha[:, None] + delta.to(tl.float32)
            tl.store(out_ptrs, out_new.to(dtype_in), mask=out_mask)

        l_i = l_new
        m_i = m_new

    # Normalize by l_i
    for d0 in tl.range(0, HEAD_DIM, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        out_ptrs = o_head_ptr + (offs_m[:, None] * stride_os + offs_d[None, :] * stride_od)
        out_mask = (m_mask[:, None] & (offs_d[None, :] < HEAD_DIM))
        out_vals = tl.load(out_ptrs, mask=out_mask, other=0.0).to(tl.float32)
        out_vals = out_vals / l_i[:, None]
        tl.store(out_ptrs, out_vals.to(dtype_in), mask=out_mask)


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Fused scaled dot-product attention implemented in a single Triton kernel.
    Computes softmax(Q @ K^T / sqrt(D)) @ V with online softmax for stability.

    Args:
        Q, K, V: [B, H, S, D] tensors, cuda, dtype in {bf16, fp16}

    Returns:
        O: [B, H, S, D], same dtype/device as inputs
    """
    # Basic validations and output allocation
    assert Q.device.type == "cuda" and K.device.type == "cuda" and V.device.type == "cuda"
    assert Q.shape == K.shape == V.shape
    assert Q.ndim == 4
    assert Q.dtype in (torch.bfloat16, torch.float16)
    assert K.dtype == Q.dtype and V.dtype == Q.dtype

    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    # Allocate output
    O = torch.zeros_like(Q)

    # Strides
    stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
    stride_ob, stride_oh, stride_os, stride_od = O.stride()

    # Launch grid: blocks over sequence (M dimension) and over batch*heads
    def grid(meta):
        return (triton.cdiv(S, meta["BLOCK_M"]), B * H)

    _sdpa_kernel[grid](
        Q, K, V, O,
        scale,
        B, H, S, D,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        IS_BF16=int(Q.dtype == torch.bfloat16),
        HEAD_DIM=D,
    )
    return O