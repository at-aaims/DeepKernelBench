import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 128, 'BLOCK_DV': 64,  'V_GROUP': 2}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 128, 'BLOCK_DV': 64,  'V_GROUP': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 64,  'V_GROUP': 2}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_DK': 64,  'BLOCK_DV': 64,  'V_GROUP': 2}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 128, 'V_GROUP': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_DK': 128, 'BLOCK_DV': 64,  'V_GROUP': 2}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 128, 'BLOCK_DV': 128, 'V_GROUP': 2}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_DK': 128, 'BLOCK_DV': 128, 'V_GROUP': 2}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 128, 'BLOCK_DV': 64,  'V_GROUP': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_DK': 128, 'BLOCK_DV': 64,  'V_GROUP': 2}, num_warps=8, num_stages=4),
    ],
    key=['S', 'D'],
)
@triton.jit
def _sdpa_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    V_GROUP: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    b = pid_bh // H
    h = pid_bh % H

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    q_base = b * stride_qb + h * stride_qh
    k_base = b * stride_kb + h * stride_kh
    v_base = b * stride_vb + h * stride_vh
    o_base = b * stride_ob + h * stride_oh

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    n_tiles = tl.cdiv(S, BLOCK_N)
    dk_tiles = tl.cdiv(D, BLOCK_DK)
    dv_tiles = tl.cdiv(D, BLOCK_DV)

    # Pass 1: compute softmax statistics m_i and l_i
    for nn in range(0, n_tiles):
        start_n = nn * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for dk in range(0, dk_tiles):
            d0 = dk * BLOCK_DK
            offs_dk = d0 + tl.arange(0, BLOCK_DK)
            mask_dk = offs_dk < D

            q_ptrs = q_ptr + q_base + offs_m[:, None] * stride_qs + offs_dk[None, :] * stride_qd
            k_ptrs = k_ptr + k_base + offs_n[None, :] * stride_ks + offs_dk[:, None] * stride_kd

            q_chunk = tl.load(q_ptrs, mask=mask_m[:, None] & mask_dk[None, :], other=0.0, eviction_policy='evict_last')
            k_chunk = tl.load(k_ptrs, mask=mask_dk[:, None] & mask_n[None, :], other=0.0, eviction_policy='evict_first')

            qk += tl.dot(q_chunk, k_chunk, out_dtype=tl.float32)

        qk = qk * sm_scale
        qk = tl.where(mask_n[None, :], qk, -float("inf"))

        m_tile = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_tile)
        p = tl.exp(qk - m_new[:, None])
        l_tile = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + l_tile
        m_i = m_new

    inv_l_i = 1.0 / l_i
    v_dtype = v_ptr.dtype.element_ty

    # Pass 2: compute output using softmax normalization with grouped V tiles
    dv_groups = tl.cdiv(dv_tiles, V_GROUP)
    for g in range(0, dv_groups):
        # Initialize accumulators for the group
        if V_GROUP == 1:
            dv0_0 = (g * V_GROUP + 0) * BLOCK_DV
            offs_dv_0 = dv0_0 + tl.arange(0, BLOCK_DV)
            mask_dv_0 = offs_dv_0 < D
            acc0 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        elif V_GROUP == 2:
            dv0_0 = (g * V_GROUP + 0) * BLOCK_DV
            dv0_1 = (g * V_GROUP + 1) * BLOCK_DV
            offs_dv_0 = dv0_0 + tl.arange(0, BLOCK_DV)
            offs_dv_1 = dv0_1 + tl.arange(0, BLOCK_DV)
            mask_dv_0 = offs_dv_0 < D
            mask_dv_1 = offs_dv_1 < D
            acc0 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
            acc1 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        else:
            dv0_0 = (g * V_GROUP + 0) * BLOCK_DV
            dv0_1 = (g * V_GROUP + 1) * BLOCK_DV
            dv0_2 = (g * V_GROUP + 2) * BLOCK_DV
            dv0_3 = (g * V_GROUP + 3) * BLOCK_DV
            offs_dv_0 = dv0_0 + tl.arange(0, BLOCK_DV)
            offs_dv_1 = dv0_1 + tl.arange(0, BLOCK_DV)
            offs_dv_2 = dv0_2 + tl.arange(0, BLOCK_DV)
            offs_dv_3 = dv0_3 + tl.arange(0, BLOCK_DV)
            mask_dv_0 = offs_dv_0 < D
            mask_dv_1 = offs_dv_1 < D
            mask_dv_2 = offs_dv_2 < D
            mask_dv_3 = offs_dv_3 < D
            acc0 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
            acc1 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
            acc3 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

        # Iterate over sequence tiles
        for nn in range(0, n_tiles):
            start_n = nn * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S

            qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for dk in range(0, dk_tiles):
                d0 = dk * BLOCK_DK
                offs_dk = d0 + tl.arange(0, BLOCK_DK)
                mask_dk = offs_dk < D

                q_ptrs = q_ptr + q_base + offs_m[:, None] * stride_qs + offs_dk[None, :] * stride_qd
                k_ptrs = k_ptr + k_base + offs_n[None, :] * stride_ks + offs_dk[:, None] * stride_kd

                q_chunk = tl.load(q_ptrs, mask=mask_m[:, None] & mask_dk[None, :], other=0.0, eviction_policy='evict_last')
                k_chunk = tl.load(k_ptrs, mask=mask_dk[:, None] & mask_n[None, :], other=0.0, eviction_policy='evict_first')
                qk += tl.dot(q_chunk, k_chunk, out_dtype=tl.float32)

            qk = qk * sm_scale
            qk = tl.where(mask_n[None, :], qk, -float("inf"))
            p = tl.exp(qk - m_i[:, None]) * inv_l_i[:, None]
            p_v = p.to(v_dtype)

            if V_GROUP == 1:
                v_ptrs0 = v_ptr + v_base + offs_n[:, None] * stride_vs + offs_dv_0[None, :] * stride_vd
                v_tile0 = tl.load(v_ptrs0, mask=mask_n[:, None] & mask_dv_0[None, :], other=0.0, eviction_policy='evict_first')
                acc0 += tl.dot(p_v, v_tile0, out_dtype=tl.float32)
            elif V_GROUP == 2:
                v_ptrs0 = v_ptr + v_base + offs_n[:, None] * stride_vs + offs_dv_0[None, :] * stride_vd
                v_ptrs1 = v_ptr + v_base + offs_n[:, None] * stride_vs + offs_dv_1[None, :] * stride_vd
                v_tile0 = tl.load(v_ptrs0, mask=mask_n[:, None] & mask_dv_0[None, :], other=0.0, eviction_policy='evict_first')
                v_tile1 = tl.load(v_ptrs1, mask=mask_n[:, None] & mask_dv_1[None, :], other=0.0, eviction_policy='evict_first')
                acc0 += tl.dot(p_v, v_tile0, out_dtype=tl.float32)
                acc1 += tl.dot(p_v, v_tile1, out_dtype=tl.float32)
            else:
                v_ptrs0 = v_ptr + v_base + offs_n[:, None] * stride_vs + offs_dv_0[None, :] * stride_vd
                v_ptrs1 = v_ptr + v_base + offs_n[:, None] * stride_vs + offs_dv_1[None, :] * stride_vd
                v_ptrs2 = v_ptr + v_base + offs_n[:, None] * stride_vs + offs_dv_2[None, :] * stride_vd
                v_ptrs3 = v_ptr + v_base + offs_n[:, None] * stride_vs + offs_dv_3[None, :] * stride_vd
                v_tile0 = tl.load(v_ptrs0, mask=mask_n[:, None] & mask_dv_0[None, :], other=0.0, eviction_policy='evict_first')
                v_tile1 = tl.load(v_ptrs1, mask=mask_n[:, None] & mask_dv_1[None, :], other=0.0, eviction_policy='evict_first')
                v_tile2 = tl.load(v_ptrs2, mask=mask_n[:, None] & mask_dv_2[None, :], other=0.0, eviction_policy='evict_first')
                v_tile3 = tl.load(v_ptrs3, mask=mask_n[:, None] & mask_dv_3[None, :], other=0.0, eviction_policy='evict_first')
                acc0 += tl.dot(p_v, v_tile0, out_dtype=tl.float32)
                acc1 += tl.dot(p_v, v_tile1, out_dtype=tl.float32)
                acc2 += tl.dot(p_v, v_tile2, out_dtype=tl.float32)
                acc3 += tl.dot(p_v, v_tile3, out_dtype=tl.float32)

        # Store results for the group
        if V_GROUP == 1:
            o_ptrs0 = o_ptr + o_base + offs_m[:, None] * stride_os + offs_dv_0[None, :] * stride_od
            tl.store(o_ptrs0, acc0.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_dv_0[None, :])
        elif V_GROUP == 2:
            o_ptrs0 = o_ptr + o_base + offs_m[:, None] * stride_os + offs_dv_0[None, :] * stride_od
            o_ptrs1 = o_ptr + o_base + offs_m[:, None] * stride_os + offs_dv_1[None, :] * stride_od
            tl.store(o_ptrs0, acc0.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_dv_0[None, :])
            tl.store(o_ptrs1, acc1.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_dv_1[None, :])
        else:
            o_ptrs0 = o_ptr + o_base + offs_m[:, None] * stride_os + offs_dv_0[None, :] * stride_od
            o_ptrs1 = o_ptr + o_base + offs_m[:, None] * stride_os + offs_dv_1[None, :] * stride_od
            o_ptrs2 = o_ptr + o_base + offs_m[:, None] * stride_os + offs_dv_2[None, :] * stride_od
            o_ptrs3 = o_ptr + o_base + offs_m[:, None] * stride_os + offs_dv_3[None, :] * stride_od
            tl.store(o_ptrs0, acc0.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_dv_0[None, :])
            tl.store(o_ptrs1, acc1.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_dv_1[None, :])
            tl.store(o_ptrs2, acc2.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_dv_2[None, :])
            tl.store(o_ptrs3, acc3.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_dv_3[None, :])


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert isinstance(Q, torch.Tensor) and isinstance(K, torch.Tensor) and isinstance(V, torch.Tensor)
    assert Q.device.type == "cuda" and K.device.type == "cuda" and V.device.type == "cuda"
    assert Q.dtype in (torch.float16, torch.bfloat16)
    assert K.dtype == Q.dtype and V.dtype == Q.dtype
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D) and V.shape == (B, H, S, D)

    O = torch.empty_like(Q)
    sm_scale = 1.0 / math.sqrt(float(D))

    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    _sdpa_fwd_kernel[grid](
        Q, K, V, O,
        B, H, S, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        sm_scale,
    )
    return O