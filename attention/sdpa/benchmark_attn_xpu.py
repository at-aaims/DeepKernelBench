import os
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F
import argparse, importlib

parent_dir = ".."
sys.path.append(parent_dir)
from reference_attn import check_torch_sdpa 


def get_flops(ngpus, batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    s = ngpus * seqlen 
    f = 4 * batch * s**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def run_benchmark(batch_size, seqlen, num_heads, head_dim,
                  causal=True, forward_only=True,
                  f=F.scaled_dot_product_attention,
                  backend=SDPBackend.FLASH_ATTENTION,
                  warmup_iter=30, num_iter=200,
                  log=True, profile=False):
    is_causal = bool(causal)
    dtype = torch.bfloat16
    device = torch.device("xpu:0")
    torch.xpu.set_device(device)

    sdpa_backend = SDPBackend(backend)

    assert head_dim % 8 == 0

    try:
        check_torch_sdpa(batch_size, seqlen, num_heads, head_dim, is_causal, sdpa_backend, device, dtype)
    except Exception as e:
        print('--------------------------------------------------------------------------------')
        print('Exceptions raised during correctness check:'                                     )
        print(e)
        print('--------------------------------------------------------------------------------')

    torch.xpu.empty_cache()

    q = torch.randn(
        batch_size,
        num_heads,
        seqlen,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        num_heads,
        seqlen,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        num_heads,
        seqlen,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    dout = torch.randn(
        batch_size, num_heads, seqlen, head_dim, device=device, dtype=dtype
    )

    try:
        q.grad = None
        k.grad = None
        v.grad = None
        with sdpa_kernel(sdpa_backend):
          out = f(
              q,
              k,
              v,
              is_causal=is_causal,
              dropout_p=0,
          )
        out.backward(dout)
    except Exception as e:
        print(e)
        return 0

    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
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
        with sdpa_kernel(sdpa_backend):
          out = f(
              q,
              k,
              v,
              is_causal=is_causal,
              dropout_p=0,
          )
        out.backward(dout)

    torch.xpu.synchronize(device=device)

    if profile:
        profiler.start()

    begin = torch.xpu.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                with sdpa_kernel(sdpa_backend):
                    _ = f(
                        q,
                        k,
                        v,
                        is_causal=is_causal,
                        dropout_p=0,
                    )
                if profile:
                    profiler.step()

    else:
        for _ in range(num_iter):
            q.grad = None
            k.grad = None
            v.grad = None
            with sdpa_kernel(sdpa_backend):
                out = f(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    dropout_p=0,
                )
            out.backward(dout)
            if profile:
                profiler.step()

    end = torch.xpu.Event(enable_timing=True)
    end.record()
    torch.xpu.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if profile:
        profiler.stop()

    if forward_only:
        flops = get_flops(1, batch_size, seqlen, num_heads, head_dim, is_causal, 'fwd')
    else:
        flops = get_flops(1, batch_size, seqlen, num_heads, head_dim, is_causal, 'fwd_bwd')
    TFLOPS = flops / (time/num_iter) / 1e12 

    print(f"{num_iter / time:.6f} iter/s, {time:.3f} sec, {TFLOPS:.1f} TFLOPS")
    return TFLOPS

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse model configuration arguments.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training or inference.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for input data.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--causal", action='store_true', help="Enable causal attention masking.")
    parser.add_argument("--forward_only", action='store_true', help="Benchmark forward pass only.")
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

    f = F.scaled_dot_product_attention

    for backend in [ SDPBackend.MATH,
                     SDPBackend.FLASH_ATTENTION,
                     SDPBackend.EFFICIENT_ATTENTION,
                     SDPBackend.CUDNN_ATTENTION
    ]:
        torch.xpu.empty_cache()
        run_benchmark(
           batch_size, seq_length, num_heads, head_dim,
           causal, forward_only, f, backend, num_iter=num_iter,
           log=True, profile=profile
        )
