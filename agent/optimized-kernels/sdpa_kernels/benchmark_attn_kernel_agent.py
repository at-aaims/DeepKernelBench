# Reference
# https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
#
import argparse
import math
import pandas as pd
import pickle
import torch
from einops import rearrange

import triton
import triton.language as tl

import os

from triton.tools.tensor_descriptor import TensorDescriptor

#from optimized_kernel_beam_search import kernel_function as kernel_function_r5
#from optimized_kernel_beam_search_r10 import kernel_function as kernel_function_r10
#from optimized_kernel_beam_search_r20 import kernel_function as kernel_function_r20
from optimized_kernel_amd import kernel_function as kernel_function_r5
from optimized_kernel_amd_r10 import kernel_function as kernel_function_r10
from optimized_kernel_amd_r20 import kernel_function as kernel_function_r20

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_flops(ngpus, batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    s = ngpus * seqlen 
    f = 4 * batch * s**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


repeats = 200
device = 'cuda'

dtype = torch.bfloat16
#dtype = torch.float16

def check_attention(B, H, N_CTX, HEAD_DIM, f, causal, dtype):
    torch.manual_seed(20)
    q = torch.empty((B, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k = torch.empty((B, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    v = torch.empty((B, H, N_CTX, HEAD_DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    sm_scale = HEAD_DIM**(-0.5)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=device))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = f(q, k, v)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)


def run_benchmark(batch_size, seqlen, num_heads, head_dim, 
                  f, causal=False, forward_only=True,
                  warmup_iter=1000, num_iter=1000,
                  log=True, profile=False):

    dtype = torch.bfloat16
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    # check correctness of Triton attention
    try:
        check_attention(batch_size, num_heads, seqlen, head_dim, f, causal, dtype=dtype)
    except Exception as e:
        print('--------------------------------------------------------------------------------')
        print('Exceptions raised during correctness check:'                                     )
        print(e)
        print('--------------------------------------------------------------------------------')
        
    torch.cuda.empty_cache()

    q, k, v = [torch.randn(batch_size, num_heads, seqlen, head_dim, device=device, dtype=dtype,
               requires_grad=True) for _ in range(3)]

    dout = torch.randn(
        batch_size, num_heads, seqlen, head_dim, device=device, dtype=dtype
    )

    sm_scale = head_dim ** (-0.5)

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
        out = f(q, k, v)

    torch.cuda.synchronize(device=device)

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = f(q, k, v)
                if profile:
                    profiler.step()

    else:
        for _ in range(num_iter):
            q.grad = None
            k.grad = None
            v.grad = None
            out = f(q, k, v)
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

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training or inference.")
    parser.add_argument("--seq_length", type=int, default=16384, help="Sequence length for input data.")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads.")
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
        #kernel_function, # CUDA Exceptions raised during correctness check: out of resource: shared memory, Required: 147456, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.
        kernel_function_r5,
        kernel_function_r10,
        kernel_function_r20,
    ]:
        torch.cuda.empty_cache()
        print(f"batch: {batch_size} seqlen: {seq_length} nhead: {num_heads} head dim {head_dim} {f.__name__}")
        run_benchmark(
           batch_size, seq_length, num_heads, head_dim,
           f, causal, forward_only,
           num_iter=num_iter,
           log=True, profile=profile
        )
