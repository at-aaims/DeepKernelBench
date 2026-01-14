import math
import torch
from einops import rearrange, repeat
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
except Exception as e:
    print(e)

# https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if softcap > 0:
        scores = scores / softcap
        scores = scores.tanh()
        scores = scores * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        key_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        reorder_ops=reorder_ops,
    )

def check_flash_attn(b, s, h, d, causal, device='cuda:0', dtype=torch.bfloat16):
    torch.manual_seed(20)
    q = (torch.empty((b, s, h, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((b, s, h, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((b, s, h, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    dout = torch.randn_like(q)
    ref_out, _ = attention_ref(q, k, v, causal=causal)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # FA implementation
    fa_out = flash_attn_func(q, k, v, causal=causal)
    fa_out.backward(dout)
    fa_dv, v.grad = v.grad.clone(), None
    fa_dk, k.grad = k.grad.clone(), None
    fa_dq, q.grad = q.grad.clone(), None
    # compare
    torch.testing.assert_close(ref_out, fa_out, atol=1e-2, rtol=0)
    rtol = 1e-2
    torch.testing.assert_close(ref_dv, fa_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(ref_dk, fa_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(ref_dq, fa_dq, atol=1e-2, rtol=rtol)

def check_flash_attn_qkvpacked(b, s, h, d, causal, device='cuda:0', dtype=torch.bfloat16):
    torch.manual_seed(20)
    qkv = torch.randn(b, s, 3, h, d, device=device, dtype=dtype, requires_grad=True)
    ref_out, _ = attention_qkvpacked_ref(qkv, causal=causal)
    g = torch.randn_like(ref_out)
    (dout,) = torch.autograd.grad(ref_out, qkv, g)
    ref_dv = dout[:,:,0]
    ref_dk = dout[:,:,1]
    ref_dq = dout[:,:,2]
    # FA implementation
    fa_out = flash_attn_qkvpacked_func(qkv, causal=causal)
    (dout,) = torch.autograd.grad(fa_out, qkv, g)
    fa_dv = dout[:,:,0]
    fa_dk = dout[:,:,1]
    fa_dq = dout[:,:,2]
    # compare
    torch.testing.assert_close(ref_out, fa_out, atol=1e-2, rtol=0)
    rtol = 1e-2
    torch.testing.assert_close(ref_dv, fa_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(ref_dk, fa_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(ref_dq, fa_dq, atol=1e-2, rtol=rtol)

def check_torch_sdpa(b, s, h, d, causal, sdpa_backend, device='cuda:0', dtype=torch.bfloat16):
    torch.manual_seed(20)
    q = (torch.empty((b, s, h, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((b, s, h, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((b, s, h, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    dout = torch.randn_like(q)
    ref_out, _ = attention_ref(q, k, v, causal=causal)
    ref_out.backward(dout)

    # SDPA implementation
    q2 = q.transpose(1,2)
    k2 = k.transpose(1,2)
    v2 = v.transpose(1,2)

    with sdpa_kernel(sdpa_backend):
        sdpa_out = F.scaled_dot_product_attention(
              q2,
              k2,
              v2,
              is_causal=causal,
              dropout_p=0,
          )
        spda_dout = torch.randn_like(sdpa_out)
        sdpa_out.backward(spda_dout)

    # compare
    torch.testing.assert_close(ref_out, sdpa_out.transpose(1,2), atol=1e-2, rtol=0)

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def check_torch_flexattn(b, s, h, d, causal, device='cuda:0', dtype=torch.bfloat16):
    torch.manual_seed(20)
    q = (torch.empty((b, s, h, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((b, s, h, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((b, s, h, d), dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
    dout = torch.randn_like(q)
    ref_out, _ = attention_ref(q, k, v, causal=causal)
    ref_out.backward(dout)

    # FA implementation
    q2 = q.transpose(1,2)
    k2 = k.transpose(1,2)
    v2 = v.transpose(1,2)
    
    block_mask = create_block_mask(
        causal_mask, B=None, H=None, Q_LEN=s, KV_LEN=s, device=device, _compile=True
    )

    fa_out = torch.compile(flex_attention)(q2, k2, v2, block_mask=block_mask)
    fa_dout = torch.randn_like(fa_out)
    fa_out.backward(fa_dout)
    # compare
    torch.testing.assert_close(ref_out, fa_out.transpose(1,2), atol=1e-2, rtol=0)
