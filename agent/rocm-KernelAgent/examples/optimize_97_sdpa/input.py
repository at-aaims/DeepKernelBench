import math
import torch
import triton
import triton.language as tl


@triton.jit
def _sdpa_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,   # number of queries per program (rows of Q / rows of output)
    BLOCK_N: tl.constexpr,   # number of keys/values per block (columns of K^T / rows of V)
    BLOCK_DK: tl.constexpr,  # tile size along head_dim for Q/K
    BLOCK_DV: tl.constexpr   # tile size along head_dim for V / output
):
    # Program IDs
    pid_m = tl.program_id(0)       # which block of query rows
    pid_bh = tl.program_id(1)      # which (batch, head) pair
    b = pid_bh // H
    h = pid_bh % H

    # Offsets for the M block (query rows)
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    # Compute base offsets for (b, h)
    q_base = b * stride_qb + h * stride_qh
    k_base = b * stride_kb + h * stride_kh
    v_base = b * stride_vb + h * stride_vh
    o_base = b * stride_ob + h * stride_oh

    # Pass 1: compute per-row max (m_i) and normalization factor (l_i) using online softmax over all keys N
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    n_tiles = tl.cdiv(S, BLOCK_N)
    dk_tiles = tl.cdiv(D, BLOCK_DK)

    for nn in range(0, n_tiles):
        start_n = nn * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # Accumulate QK^T for this (BLOCK_M x BLOCK_N) tile
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for dk in range(0, dk_tiles):
            d0 = dk * BLOCK_DK
            offs_dk = d0 + tl.arange(0, BLOCK_DK)
            mask_dk = offs_dk < D

            # Load Q chunk [BLOCK_M, BLOCK_DK]
            q_ptrs = q_ptr + q_base \
                     + offs_m[:, None] * stride_qs \
                     + offs_dk[None, :] * stride_qd
            q_chunk = tl.load(q_ptrs, mask=mask_m[:, None] & mask_dk[None, :], other=0.0)
            q_chunk = q_chunk.to(tl.float32)

            # Load K chunk transposed as [BLOCK_DK, BLOCK_N]
            k_ptrs = k_ptr + k_base \
                     + offs_n[None, :] * stride_ks \
                     + offs_dk[:, None] * stride_kd
            k_chunk_t = tl.load(k_ptrs, mask=mask_dk[:, None] & mask_n[None, :], other=0.0)
            k_chunk_t = k_chunk_t.to(tl.float32)

            # Accumulate dot: [BLOCK_M, BLOCK_DK] x [BLOCK_DK, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            qk += tl.dot(q_chunk, k_chunk_t)

        # Scale logits
        qk = qk * sm_scale

        # Row-wise max for this tile
        m_tile = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_tile)
        # Compute exp(qk - m_new) and update l_i with the online formula
        p = tl.exp(qk - m_new[:, None])
        l_tile = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + l_tile
        m_i = m_new

    # Pass 2: use the computed m_i and l_i to form normalized probabilities and compute output = P @ V
    dv_tiles = tl.cdiv(D, BLOCK_DV)

    for dv in range(0, dv_tiles):
        dv0 = dv * BLOCK_DV
        offs_dv = dv0 + tl.arange(0, BLOCK_DV)
        mask_dv = offs_dv < D

        # Accumulator for this output slice [BLOCK_M, BLOCK_DV]
        acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)

        for nn in range(0, n_tiles):
            start_n = nn * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S

            # Recompute QK^T tile (same as in pass 1)
            qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for dk in range(0, dk_tiles):
                d0 = dk * BLOCK_DK
                offs_dk = d0 + tl.arange(0, BLOCK_DK)
                mask_dk = offs_dk < D

                q_ptrs = q_ptr + q_base \
                         + offs_m[:, None] * stride_qs \
                         + offs_dk[None, :] * stride_qd
                q_chunk = tl.load(q_ptrs, mask=mask_m[:, None] & mask_dk[None, :], other=0.0)
                q_chunk = q_chunk.to(tl.float32)

                k_ptrs = k_ptr + k_base \
                         + offs_n[None, :] * stride_ks \
                         + offs_dk[:, None] * stride_kd
                k_chunk_t = tl.load(k_ptrs, mask=mask_dk[:, None] & mask_n[None, :], other=0.0)
                k_chunk_t = k_chunk_t.to(tl.float32)

                qk += tl.dot(q_chunk, k_chunk_t)

            qk = qk * sm_scale
            # Compute normalized probabilities: p_norm = exp(qk - m_i) / l_i
            p = tl.exp(qk - m_i[:, None])
            p_norm = p / l_i[:, None]

            # Load V tile [BLOCK_N, BLOCK_DV] and accumulate acc += p_norm @ V_tile
            v_ptrs = v_ptr + v_base \
                     + offs_n[:, None] * stride_vs \
                     + offs_dv[None, :] * stride_vd
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
            v_tile = v_tile.to(tl.float32)

            acc += tl.dot(p_norm, v_tile)

        # Store the [BLOCK_M, BLOCK_DV] slice to output
        o_ptrs = o_ptr + o_base \
                 + offs_m[:, None] * stride_os \
                 + offs_dv[None, :] * stride_od
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_dv[None, :])


def kernel_function(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention using a single fused Triton kernel.

    What is fused:
    - This kernel fuses the full forward pass of SDPA for each (batch, head) and a block of query positions:
      1) QK^T matmul to produce attention logits
      2) Numerically-stable softmax (two-pass: first pass computes per-row maxima and normalization factors)
      3) Matmul with V to form the output
    - All math is performed inside the Triton kernel; the Python wrapper only validates inputs, allocates output,
      prepares strides/scales, and launches the kernel.

    Note on two-pass strategy inside the single kernel:
    - To support very large head dimensions (e.g., 1024) without excessive register pressure, we first sweep over
      all key blocks to compute the per-row max and normalizer (l_i) using an online softmax formulation.
    - We then sweep again to form normalized probabilities and accumulate the output with V in tiles of the head
      dimension. This keeps the compute fused within one kernel while controlling on-chip memory usage.

    AMD ROCm considerations:
    - BLOCK sizes are chosen as multiples of 64 to match AMD wavefronts (64 lanes).
    - num_warps is selected as 4 or 8 in the launch parameters for good occupancy on AMD GPUs.
    """
    # Argument validation and setup; no math here.
    assert isinstance(Q, torch.Tensor) and isinstance(K, torch.Tensor) and isinstance(V, torch.Tensor)
    assert Q.device.type == "cuda" and K.device.type == "cuda" and V.device.type == "cuda", "Inputs must be CUDA tensors"
    assert Q.dtype in (torch.float16, torch.bfloat16), "Only fp16/bf16 supported"
    assert K.dtype == Q.dtype and V.dtype == Q.dtype, "Q, K, V must have same dtype"
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4, "Q, K, V must be 4D [B, H, S, D]"
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D) and V.shape == (B, H, S, D), "Q, K, V shapes must match"

    # Output allocation
    O = torch.empty_like(Q)

    # Compute scale as 1/sqrt(D) in float32 for numerical stability
    sm_scale = 1.0 / math.sqrt(float(D))

    # Choose block sizes tailored for AMD wavefront (64)
    BLOCK_M = 64       # query rows per block
    BLOCK_N = 64       # key/val rows per block
    BLOCK_DK = 64      # inner head-dim tile for Q/K
    BLOCK_DV = 64      # head-dim tile for V/output

    # Grid: one program per (block of S) x (B*H)
    grid = (triton.cdiv(S, BLOCK_M), B * H)

    # Launch kernel
    _sdpa_fwd_kernel[grid](
        Q, K, V, O,
        B, H, S, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DK=BLOCK_DK,
        BLOCK_DV=BLOCK_DV,
        num_warps=8,       # multiples of AMD wavefronts; 8*64 = 512 lanes
        num_stages=2       # simple pipelining
    )
    return O