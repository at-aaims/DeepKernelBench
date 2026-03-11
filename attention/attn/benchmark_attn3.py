import argparse
import os
import sys
import torch

try:
    from flash_attn_interface import flash_attn_func
except ImportError as e:
    print(f"Error: Required library is not installed or available. {e}")
    print("Please install FA3 to run this script.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

parent_dir = ".."
sys.path.append(parent_dir)
from reference_attn import attention_ref

def get_flops(ngpus, batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    s = ngpus * seqlen
    f = 4 * batch * s**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def run_benchmark(batch_size, seqlen, num_heads, head_dim,
                  causal=False, forward_only=False,
                  f=flash_attn_func, warmup_iter=1000, num_iter=1000,
                  log=True, profile=False):
    dtype = torch.bfloat16
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)
    deterministic = False

    assert head_dim % 8 == 0

    try:
        b = batch_size
        s = seqlen
        h = num_heads
        d = head_dim
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
        fa_out = f(q, k, v, causal=causal)
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

    except Exception as e:
        print('--------------------------------------------------------------------------------')
        print('Exceptions raised during correctness check:'                                     )
        print(e)
        print('--------------------------------------------------------------------------------')

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


    for _ in range(warmup_iter):
        q.grad = None
        k.grad = None
        v.grad = None
        out = f(
            q,
            k,
            v,
            causal=causal,
            window_size=(-1, -1),
            deterministic=deterministic,
            return_attn_probs=False,
        )
        out.backward(dout)

    torch.cuda.synchronize(device=device)

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                f(q,
                  k,
                  v,
                  causal=causal,
                  window_size=(-1, -1),
                  deterministic=deterministic,
                  return_attn_probs=False,
                )
                if profile:
                    profiler.step()

    else:
        for _ in range(num_iter):
            q.grad = None
            k.grad = None
            v.grad = None
            out = f(
                q,
                k,
                v,
                causal=causal,
                window_size=(-1, -1),
                deterministic=deterministic,
                return_attn_probs=False,
            )
            out.backward(dout)
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

    for f in [
        flash_attn_func,
    ]:
        torch.cuda.empty_cache()
        #if rank == 0:
            #print(f"# {f.__name__}")
        run_benchmark(
           batch_size, seq_length, num_heads, head_dim,
           causal, forward_only,
           f, num_iter=num_iter,
           log=True, profile=profile
        )
