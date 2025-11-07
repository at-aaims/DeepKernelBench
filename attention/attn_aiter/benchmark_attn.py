import os
import sys
import torch
import argparse
import aiter
from aiter.test_mha_common import (
    attention_ref,
    attn_bias_from_alibi_slopes,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax,
)

def get_flops(ngpus, batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    s = ngpus * seqlen 
    f = 4 * batch * s**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def run_torch(
    q,
    k,
    v,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window,
    upcast=True,
    reorder_ops=False,
    query_padding_mask=None,
    key_padding_mask=None,
):
    (_, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if bias is not None:
        attn_bias = bias
    elif alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, causal=causal
        )
    else:
        attn_bias = None

    out, _, softmax_lse = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
        upcast=upcast,
        reorder_ops=reorder_ops,
    )

    if dout is None:
        return out, softmax_lse
    elif bias is not None:
        dq, dk, dv, dbias = torch.autograd.grad(out, (q, k, v, bias), dout)
        # If seqlen_q > seqlen_k with mask, pytorch will output NaN.
        # Align with ck behavior here
        dbias = torch.nan_to_num(dbias, nan=0.0)
        return out, softmax_lse, dq, dk, dv, dbias
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, softmax_lse, dq, dk, dv, None

def run_ck(
    q,
    k,
    v,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    deterministic=False,
    return_lse=True,
    return_attn_probs=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
):
    (out, softmax_lse, S_dmask) = aiter.flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        None,  # softmax_scale
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        cu_seqlens_q,
        cu_seqlens_kv
    )

    if dropout_p > 0.0:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, d) = k.shape
        (_, seqlen_k, _, d_v) = v.shape
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            None,
            None,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
    else:
        dropout_mask = None

    if dout is None:
        return out, softmax_lse, dropout_mask
    elif bias is not None:
        (dq, dk, dv, dbias) = torch.autograd.grad(
            out,
            (q, k, v, bias),
            dout,
            retain_graph=True
        )
        return out, softmax_lse, dropout_mask, dq, dk, dv, dbias
    else:
        (dq, dk, dv) = torch.autograd.grad(
            out,
            (q, k, v),
            dout,
            retain_graph=True
        )
        return out, softmax_lse, dropout_mask, dq, dk, dv, None



def run_benchmark(batch_size, seqlen, num_heads, head_dim, 
                  causal=True, forward_only=False,
                  warmup_iter=1000, num_iter=1000,
                  log=True, profile=False):
    dtype = torch.bfloat16
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    deterministic = False
    attn_bias = None
    alibi_slopes = None
    dropout_p = 0.0
    window_size = (-1,-1)

    return_lse = True
    return_attn_probs = True
 
    assert head_dim % 8 == 0
    torch.cuda.empty_cache()

    q = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    dout = torch.randn(
        batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype
    )

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=3,
                active=5,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(
                    f"./benchmark/logs/{f.__name__}", f"rank_{dist.get_rank()}"
                )
            ),
        )

    # Correctness check
    out, softmax_lse, dropout_mask, dq, dk, dv, dbias = run_ck(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        causal,
        window_size,
        deterministic,
        return_lse,
        return_attn_probs,
    )

    try:
        out_ref, softmax_lse_ref, dq_ref, dk_ref, dv_ref, dbias_ref = run_torch(
            q,
            k,
            v,
            attn_bias,
            alibi_slopes,
            dout,
            dropout_p,
            dropout_mask,
            causal,
            window_size,
        )

        out_pt, softmax_lse_pt, dq_pt, dk_pt, dv_pt, dbias_pt = run_torch(
            q,
            k,
            v,
            attn_bias,
            alibi_slopes,
            dout,
            dropout_p,
            dropout_mask,
            causal,
            window_size,
            upcast=False,
            reorder_ops=True,
        )

        print(f"Output max diff: {(out - out_ref).abs().max().item()}")
        print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
        out_tol = max(2 * (out_pt - out_ref).abs().max().item(), 0.01)
        assert (out - out_ref).abs().max().item() <= out_tol

        print(f"softmax_lse max diff: {(softmax_lse - softmax_lse_ref).abs().max().item()}")
        print(
            f"softmax_lse Pytorch max diff: {(softmax_lse_pt - softmax_lse_ref).abs().max().item()}"
        )
        softmax_lse_tol = max(
            2 * (softmax_lse_pt - softmax_lse_ref).abs().max().item(), 0.01
        )
        # assert (softmax_lse - softmax_lse_ref).abs().max().item() <= softmax_lse_tol

        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")

        dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
        dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
        dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)

        assert (dq - dq_ref).abs().max().item() <= dq_tol
        assert (dk - dk_ref).abs().max().item() <= dk_tol
        assert (dv - dv_ref).abs().max().item() <= dv_tol

        if attn_bias is not None:
            print(f"dBias max diff: {(dbias - dbias_ref).abs().max().item()}")
            print(f"dBias Pytorch max diff: {(dbias_pt - dbias_ref).abs().max().item()}")
            dbias_tol = max(10 * (dbias_pt - dbias_ref).abs().max().item(), 0.01)
            assert (dbias - dbias_ref).abs().max().item() <= dbias_tol

    except Exception as e:
        print('--------------------------------------------------------------------------------')
        print('Exceptions raised during correctness check:'                                     )
        print(e)
        print('--------------------------------------------------------------------------------')

    for _ in range(warmup_iter):
        q.grad = None
        k.grad = None
        v.grad = None
        out, _ = aiter.flash_attn_func(
         q,
         k,
         v,
         dropout_p,
         None,  # softmax_scale
         causal,
         window_size,
         bias=None,
         alibi_slopes=None,
         deterministic=False,
         return_lse=True,         # assertion error on false
         return_attn_probs=False,
         cu_seqlens_q=None,
         cu_seqlens_kv=None
        )
        (dq, dk, dv) = torch.autograd.grad(
            out,
            (q, k, v),
            dout,
            retain_graph=True
        )

    torch.cuda.synchronize(device=device)

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = aiter.flash_attn_func(
                 q,
                 k,
                 v,
                 dropout_p,
                 None,  # softmax_scale
                 causal,
                 window_size,
                 bias=None,
                 alibi_slopes=None,
                 deterministic=False,
                 return_lse=False,
                 return_attn_probs=False,
                 cu_seqlens_q=None,
                 cu_seqlens_kv=None
                )
                if profile:
                    profiler.step()

    else:
        for _ in range(num_iter):
            q.grad = None
            k.grad = None
            v.grad = None
            out, _ = aiter.flash_attn_func(
             q,
             k,
             v,
             dropout_p,
             None,  # softmax_scale
             causal,
             window_size,
             bias=None,
             alibi_slopes=None,
             deterministic=False,
             return_lse=True,         # assertion error on false
             return_attn_probs=False,
             cu_seqlens_q=None,
             cu_seqlens_kv=None
            )
            (dq, dk, dv) = torch.autograd.grad(
                out,
                (q, k, v),
                dout,
                retain_graph=True
            )
            if profile:
                profiler.step()

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if profile:
        profiler.stop()

    if forward_only:
        flops = get_flops(1, batch_size, seqlen, num_heads, head_dim, causal, 'fwd')
    else:
        flops = get_flops(1, batch_size, seqlen, num_heads, head_dim, causal, 'fwd_bwd')

    TFLOPS = flops / (time/num_iter) / 1e12 

    print(f"{num_iter / time:.6f} iter/s, {time:.3f} sec, {TFLOPS:.1f} TFLOPS")
    return TFLOPS

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse model configuration arguments.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training or inference.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for input data.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--causal", action='store_true', help="Enable causal attention masking.")
    parser.add_argument("--forward_only", action='store_true', help="Benchmark forward pass only.")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--profile", action='store_true', help="Enable profiling.")

    args = parser.parse_args()
    batch_size = args.batch_size
    seq_length = args.seq_length
    num_heads = args.num_heads
    head_dim = args.head_dim
    num_iter = args.num_iter
    causal = args.causal
    forward_only = args.forward_only
    profile = args.profile

    torch.cuda.empty_cache()
    #if rank == 0:
        #print(f"# {f.__name__}")
    run_benchmark(
       batch_size, seq_length, num_heads, head_dim,
       causal, forward_only,
       num_iter=num_iter,
       log=True, profile=profile
    )

