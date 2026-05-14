import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # MI300A-oriented configs (CDNA3, wave64)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 32}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 32}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 128, 'BLOCK_DV': 128, 'GROUP_SIZE_M': 16}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 128, 'BLOCK_DV': 64,  'GROUP_SIZE_M': 32}, num_warps=4, num_stages=6),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 64}, num_warps=4, num_stages=5),
        # Fallbacks (retain good NVIDIA-like shapes that also do well on AMD)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_DK': 64,  'BLOCK_DV': 64,  'GROUP_SIZE_M': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 64,  'GROUP_SIZE_M': 8},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_DK': 64,  'BLOCK_DV': 64,  'GROUP_SIZE_M': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 64,  'GROUP_SIZE_M': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 256, 'GROUP_SIZE_M': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 256, 'GROUP_SIZE_M': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 64,  'GROUP_SIZE_M': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 64,  'GROUP_SIZE_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 128, 'BLOCK_DV': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 8},  num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 4},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_DK': 64,  'BLOCK_DV': 128, 'GROUP_SIZE_M': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_DK': 128, 'BLOCK_DV': 128, 'GROUP_SIZE_M': 16}, num_warps=8, num_stages=5),
    ],
    key=['S', 'D'],
)
@triton.heuristics({
    'EVEN_M': lambda args: args['S'] % args['BLOCK_M'] == 0,
    'EVEN_N': lambda args: args['S'] % args['BLOCK_N'] == 0,
    'EVEN_DK': lambda args: args['D'] % args['BLOCK_DK'] == 0,
    'EVEN_DV': lambda args: args['D'] % args['BLOCK_DV'] == 0,
})
@triton.jit
def _sdpa_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    SM_SCALE: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_DK: tl.constexpr,
    EVEN_DV: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_m = tl.cdiv(S, BLOCK_M)
    num_bh = B * H

    group_stride = num_bh * GROUP_SIZE_M
    group_id_m = pid // group_stride
    pid_in_group = pid % group_stride
    bh = pid_in_group // GROUP_SIZE_M
    m_in_group = pid_in_group % GROUP_SIZE_M
    pid_m = group_id_m * GROUP_SIZE_M + m_in_group
    if pid_m >= num_m:
        return

    b = bh // H
    h = bh % H

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    q_base = b * stride_qb + h * stride_qh
    k_base = b * stride_kb + h * stride_kh
    v_base = b * stride_vb + h * stride_vh
    o_base = b * stride_ob + h * stride_oh

    n_tiles = tl.cdiv(S, BLOCK_N)
    dk_tiles = tl.cdiv(D, BLOCK_DK)
    dv_tiles = tl.cdiv(D, BLOCK_DV)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)

    q_row_ptrs = q_ptr + q_base + offs_m[:, None] * stride_qs
    o_row_ptrs = o_ptr + o_base + offs_m[:, None] * stride_os

    LOG2E = 1.4426950408889634

    tl.multiple_of(offs_dk, 16)
    tl.multiple_of(offs_dv, 16)

    for nn in tl.range(0, n_tiles):
        start_n = nn * BLOCK_N
        n_idx = start_n + offs_n
        mask_n = n_idx < S

        k_col_ptrs = k_ptr + k_base + n_idx[None, :] * stride_ks

        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for dk in tl.range(0, dk_tiles):
            d0 = dk * BLOCK_DK
            dk_idx = d0 + offs_dk
            mask_dk = dk_idx < D

            if EVEN_M and EVEN_DK:
                q_chunk = tl.load(q_row_ptrs + dk_idx[None, :] * stride_qd)
            else:
                q_chunk = tl.load(
                    q_row_ptrs + dk_idx[None, :] * stride_qd,
                    mask=mask_m[:, None] & mask_dk[None, :],
                    other=0.0
                )

            if EVEN_N and EVEN_DK:
                k_chunk = tl.load(k_col_ptrs + dk_idx[:, None] * stride_kd)
            else:
                k_chunk = tl.load(
                    k_col_ptrs + dk_idx[:, None] * stride_kd,
                    mask=mask_dk[:, None] & mask_n[None, :],
                    other=0.0
                )

            qk += tl.dot(q_chunk, k_chunk)

        qk = qk * SM_SCALE

        m_tile = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_tile)

        p32 = tl.exp2((qk - m_new[:, None]) * LOG2E)
        l_tile = tl.sum(p32, axis=1)

        alpha = tl.exp2((m_i - m_new) * LOG2E)
        l_i_prev_scaled = l_i * alpha
        l_i_new = l_i_prev_scaled + l_tile
        inv_l_new = 1.0 / l_i_new
        beta = l_i_prev_scaled * inv_l_new

        p_cast = p32.to(DTYPE)

        v_col_ptrs = v_ptr + v_base + n_idx[:, None] * stride_vs

        for dv in tl.range(0, dv_tiles):
            dv0 = dv * BLOCK_DV
            dv_idx = dv0 + offs_dv
            mask_dv = dv_idx < D

            if EVEN_N and EVEN_DV:
                v_tile = tl.load(v_col_ptrs + dv_idx[None, :] * stride_vd)
            else:
                v_tile = tl.load(
                    v_col_ptrs + dv_idx[None, :] * stride_vd,
                    mask=mask_n[:, None] & mask_dv[None, :],
                    other=0.0
                )

            update = tl.dot(p_cast, v_tile)

            o_ptrs = o_row_ptrs + dv_idx[None, :] * stride_od
            if nn == 0:
                o_new = update * inv_l_new[:, None]
            else:
                if EVEN_M and EVEN_DV:
                    o_old = tl.load(o_ptrs).to(tl.float32)
                else:
                    o_old = tl.load(
                        o_ptrs,
                        mask=mask_m[:, None] & mask_dv[None, :],
                        other=0.0
                    ).to(tl.float32)
                o_new = o_old * beta[:, None] + update * inv_l_new[:, None]

            if EVEN_M and EVEN_DV:
                tl.store(o_ptrs, o_new.to(DTYPE))
            else:
                tl.store(
                    o_ptrs,
                    o_new.to(DTYPE),
                    mask=mask_m[:, None] & mask_dv[None, :]
                )

        m_i = m_new
        l_i = l_i_new


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
    dtype_triton = tl.float16 if Q.dtype == torch.float16 else tl.bfloat16

    def grid(meta):
        return (triton.cdiv(S, meta['BLOCK_M']) * B * H,)

    _sdpa_fwd_kernel[grid](
        Q, K, V, O,
        B, H, S, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        SM_SCALE=sm_scale,
        DTYPE=dtype_triton,
    )
    return O