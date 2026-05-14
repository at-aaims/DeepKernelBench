import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_DK": 128, "BLOCK_DV": 128, "GROUP_DV": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_DK": 64,  "BLOCK_DV": 128, "GROUP_DV": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_DK": 128, "BLOCK_DV": 128, "GROUP_DV": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_DK": 128, "BLOCK_DV": 128, "GROUP_DV": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_DK": 128, "BLOCK_DV": 128, "GROUP_DV": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_DK": 64,  "BLOCK_DV": 128, "GROUP_DV": 8}, num_warps=2, num_stages=2),
    ],
    key=["S", "D", "H"],
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
    GROUP_DV: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    q_base = b * stride_qb + h * stride_qh
    k_base = b * stride_kb + h * stride_kh
    v_base = b * stride_vb + h * stride_vh
    o_base = b * stride_ob + h * stride_oh

    n_tiles = tl.cdiv(S, BLOCK_N)
    dk_tiles = tl.cdiv(D, BLOCK_DK)

    dv_group_size = BLOCK_DV * GROUP_DV
    dv_groups = tl.cdiv(D, dv_group_size)

    for g in range(0, dv_groups):
        dv_group_start = g * dv_group_size

        acc0 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        acc4 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        acc5 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        acc6 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
        acc7 = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

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
                k_chunk_t = tl.load(k_ptrs, mask=mask_dk[:, None] & mask_n[None, :], other=0.0, eviction_policy='evict_last')

                qk += tl.dot(q_chunk, k_chunk_t, out_dtype=tl.float32)

            qk = qk * sm_scale
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, -float("inf"))

            m_tile = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_tile)
            p = tl.exp(qk - m_new[:, None])
            p = tl.where(mask_m[:, None], p, 0.0)
            l_tile = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_new)
            alpha = tl.where(mask_m, alpha, 1.0)

            base_v_ptrs = v_ptr + v_base + offs_n[:, None] * stride_vs

            dv0 = dv_group_start + 0 * BLOCK_DV
            offs_dv0 = dv0 + tl.arange(0, BLOCK_DV)
            mask_dv0 = offs_dv0 < D
            if GROUP_DV > 0:
                v_ptrs0 = base_v_ptrs + offs_dv0[None, :] * stride_vd
                v_tile0 = tl.load(v_ptrs0, mask=mask_n[:, None] & mask_dv0[None, :], other=0.0, eviction_policy='evict_last')
                acc0 = acc0 * alpha[:, None] + tl.dot(p.to(v_tile0.dtype), v_tile0, out_dtype=tl.float32)

            dv1 = dv_group_start + 1 * BLOCK_DV
            offs_dv1 = dv1 + tl.arange(0, BLOCK_DV)
            mask_dv1 = offs_dv1 < D
            if GROUP_DV > 1:
                v_ptrs1 = base_v_ptrs + offs_dv1[None, :] * stride_vd
                v_tile1 = tl.load(v_ptrs1, mask=mask_n[:, None] & mask_dv1[None, :], other=0.0, eviction_policy='evict_last')
                acc1 = acc1 * alpha[:, None] + tl.dot(p.to(v_tile1.dtype), v_tile1, out_dtype=tl.float32)

            dv2 = dv_group_start + 2 * BLOCK_DV
            offs_dv2 = dv2 + tl.arange(0, BLOCK_DV)
            mask_dv2 = offs_dv2 < D
            if GROUP_DV > 2:
                v_ptrs2 = base_v_ptrs + offs_dv2[None, :] * stride_vd
                v_tile2 = tl.load(v_ptrs2, mask=mask_n[:, None] & mask_dv2[None, :], other=0.0, eviction_policy='evict_last')
                acc2 = acc2 * alpha[:, None] + tl.dot(p.to(v_tile2.dtype), v_tile2, out_dtype=tl.float32)

            dv3 = dv_group_start + 3 * BLOCK_DV
            offs_dv3 = dv3 + tl.arange(0, BLOCK_DV)
            mask_dv3 = offs_dv3 < D
            if GROUP_DV > 3:
                v_ptrs3 = base_v_ptrs + offs_dv3[None, :] * stride_vd
                v_tile3 = tl.load(v_ptrs3, mask=mask_n[:, None] & mask_dv3[None, :], other=0.0, eviction_policy='evict_last')
                acc3 = acc3 * alpha[:, None] + tl.dot(p.to(v_tile3.dtype), v_tile3, out_dtype=tl.float32)

            dv4 = dv_group_start + 4 * BLOCK_DV
            offs_dv4 = dv4 + tl.arange(0, BLOCK_DV)
            mask_dv4 = offs_dv4 < D
            if GROUP_DV > 4:
                v_ptrs4 = base_v_ptrs + offs_dv4[None, :] * stride_vd
                v_tile4 = tl.load(v_ptrs4, mask=mask_n[:, None] & mask_dv4[None, :], other=0.0, eviction_policy='evict_last')
                acc4 = acc4 * alpha[:, None] + tl.dot(p.to(v_tile4.dtype), v_tile4, out_dtype=tl.float32)

            dv5 = dv_group_start + 5 * BLOCK_DV
            offs_dv5 = dv5 + tl.arange(0, BLOCK_DV)
            mask_dv5 = offs_dv5 < D
            if GROUP_DV > 5:
                v_ptrs5 = base_v_ptrs + offs_dv5[None, :] * stride_vd
                v_tile5 = tl.load(v_ptrs5, mask=mask_n[:, None] & mask_dv5[None, :], other=0.0, eviction_policy='evict_last')
                acc5 = acc5 * alpha[:, None] + tl.dot(p.to(v_tile5.dtype), v_tile5, out_dtype=tl.float32)

            dv6 = dv_group_start + 6 * BLOCK_DV
            offs_dv6 = dv6 + tl.arange(0, BLOCK_DV)
            mask_dv6 = offs_dv6 < D
            if GROUP_DV > 6:
                v_ptrs6 = base_v_ptrs + offs_dv6[None, :] * stride_vd
                v_tile6 = tl.load(v_ptrs6, mask=mask_n[:, None] & mask_dv6[None, :], other=0.0, eviction_policy='evict_last')
                acc6 = acc6 * alpha[:, None] + tl.dot(p.to(v_tile6.dtype), v_tile6, out_dtype=tl.float32)

            dv7 = dv_group_start + 7 * BLOCK_DV
            offs_dv7 = dv7 + tl.arange(0, BLOCK_DV)
            mask_dv7 = offs_dv7 < D
            if GROUP_DV > 7:
                v_ptrs7 = base_v_ptrs + offs_dv7[None, :] * stride_vd
                v_tile7 = tl.load(v_ptrs7, mask=mask_n[:, None] & mask_dv7[None, :], other=0.0, eviction_policy='evict_last')
                acc7 = acc7 * alpha[:, None] + tl.dot(p.to(v_tile7.dtype), v_tile7, out_dtype=tl.float32)

            l_i = l_i * alpha + l_tile
            m_i = m_new

        o_dtype = o_ptr.dtype.element_ty

        idxs0 = dv_group_start + 0 * BLOCK_DV + tl.arange(0, BLOCK_DV)
        if GROUP_DV > 0:
            o_ptrs0 = o_ptr + o_base + offs_m[:, None] * stride_os + idxs0[None, :] * stride_od
            tl.store(o_ptrs0, (acc0 / l_i[:, None]).to(o_dtype), mask=mask_m[:, None] & (idxs0[None, :] < D))

        idxs1 = dv_group_start + 1 * BLOCK_DV + tl.arange(0, BLOCK_DV)
        if GROUP_DV > 1:
            o_ptrs1 = o_ptr + o_base + offs_m[:, None] * stride_os + idxs1[None, :] * stride_od
            tl.store(o_ptrs1, (acc1 / l_i[:, None]).to(o_dtype), mask=mask_m[:, None] & (idxs1[None, :] < D))

        idxs2 = dv_group_start + 2 * BLOCK_DV + tl.arange(0, BLOCK_DV)
        if GROUP_DV > 2:
            o_ptrs2 = o_ptr + o_base + offs_m[:, None] * stride_os + idxs2[None, :] * stride_od
            tl.store(o_ptrs2, (acc2 / l_i[:, None]).to(o_dtype), mask=mask_m[:, None] & (idxs2[None, :] < D))

        idxs3 = dv_group_start + 3 * BLOCK_DV + tl.arange(0, BLOCK_DV)
        if GROUP_DV > 3:
            o_ptrs3 = o_ptr + o_base + offs_m[:, None] * stride_os + idxs3[None, :] * stride_od
            tl.store(o_ptrs3, (acc3 / l_i[:, None]).to(o_dtype), mask=mask_m[:, None] & (idxs3[None, :] < D))

        idxs4 = dv_group_start + 4 * BLOCK_DV + tl.arange(0, BLOCK_DV)
        if GROUP_DV > 4:
            o_ptrs4 = o_ptr + o_base + offs_m[:, None] * stride_os + idxs4[None, :] * stride_od
            tl.store(o_ptrs4, (acc4 / l_i[:, None]).to(o_dtype), mask=mask_m[:, None] & (idxs4[None, :] < D))

        idxs5 = dv_group_start + 5 * BLOCK_DV + tl.arange(0, BLOCK_DV)
        if GROUP_DV > 5:
            o_ptrs5 = o_ptr + o_base + offs_m[:, None] * stride_os + idxs5[None, :] * stride_od
            tl.store(o_ptrs5, (acc5 / l_i[:, None]).to(o_dtype), mask=mask_m[:, None] & (idxs5[None, :] < D))

        idxs6 = dv_group_start + 6 * BLOCK_DV + tl.arange(0, BLOCK_DV)
        if GROUP_DV > 6:
            o_ptrs6 = o_ptr + o_base + offs_m[:, None] * stride_os + idxs6[None, :] * stride_od
            tl.store(o_ptrs6, (acc6 / l_i[:, None]).to(o_dtype), mask=mask_m[:, None] & (idxs6[None, :] < D))

        idxs7 = dv_group_start + 7 * BLOCK_DV + tl.arange(0, BLOCK_DV)
        if GROUP_DV > 7:
            o_ptrs7 = o_ptr + o_base + offs_m[:, None] * stride_os + idxs7[None, :] * stride_od
            tl.store(o_ptrs7, (acc7 / l_i[:, None]).to(o_dtype), mask=mask_m[:, None] & (idxs7[None, :] < D))


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    assert isinstance(Q, torch.Tensor) and isinstance(K, torch.Tensor) and isinstance(V, torch.Tensor)
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dtype in (torch.float16, torch.bfloat16)
    assert K.dtype == Q.dtype and V.dtype == Q.dtype
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D) and V.shape == (B, H, S, D)

    O = torch.empty_like(Q)

    sm_scale = 1.0 / math.sqrt(float(D))

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)

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