import sys
import time
import torch
import torch.nn.functional as F
import argparse

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096

try:
    flex_attention = torch.compile(flex_attention)
except Exception as e:
    print("torch.compile(flex_attention) failed or not necessary:", e)

parent_dir = ".."
sys.path.append(parent_dir)
from reference_attn import check_torch_flexattn 

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def call_flex(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)

def call_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def get_flops(ngpus, batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    s = ngpus * seqlen 
    f = 4 * batch * s**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def run_benchmark(batch_size, seqlen, num_heads, head_dim,
                  fn=call_flex, forward_only=False, use_block_mask=False,
                  warmup_iter=1000, num_iter=1000,
                  log=True, profile=False):
    dtype = torch.bfloat16
    device = torch.device(f"xpu:0")

    torch.xpu.set_device(device)
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats(device)

    try:
        check_torch_flexattn(batch_size, seqlen, num_heads, head_dim, True, device, dtype)
    except Exception as e:
        print('--------------------------------------------------------------------------------')
        print('Exceptions raised during correctness check:'                                     )
        print(e)
        print('--------------------------------------------------------------------------------')
    
    torch.xpu.empty_cache()

    torch.manual_seed(0)

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
        batch_size,
        num_heads,
        seqlen,
        head_dim,
        device=device,
        dtype=dtype,
    )

    block_mask = create_block_mask(
        causal_mask, B=None, H=None, Q_LEN=seqlen, KV_LEN=seqlen,
        device=device, _compile=True
    )

    for i in range(warmup_iter):
        out = fn(q, k, v, block_mask) if use_block_mask else fn(q, k, v)
        out.backward(dout)
    torch.xpu.synchronize()

    begin = torch.xpu.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = fn(q, k, v, block_mask) if use_block_mask else fn(q, k, v)
    else:
        for _ in range(num_iter):
            q.grad = None
            k.grad = None
            v.grad = None
            out = fn(q, k, v, block_mask) if use_block_mask else fn(q, k, v)
            out.backward(dout)

    end = torch.xpu.Event(enable_timing=True)
    end.record()
    torch.xpu.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0
    avg_time = time/num_iter

    if forward_only:
        flops = get_flops(1, batch_size, seqlen, num_heads, head_dim, True, 'fwd')
    else:
        flops = get_flops(1, batch_size, seqlen, num_heads, head_dim, True, 'fwd_bwd')
    tflops = flops / avg_time / 1e12 

    peak_mem_gb = None
    peak_bytes = torch.xpu.max_memory_allocated(device)
    peak_mem_gb = peak_bytes / (1024**3)

    r = {
        "name": fn.__name__,
        "avg_s": avg_time,
        "tflops": tflops,
        "peak_mem_gb": peak_mem_gb,
    }
    print(f"{r['name']}:")
    print(f"  average time     : {r['avg_s'] * 1000:.3f} ms")
    print(f"  TFLOPS   : {r['tflops']}")
    if r["peak_mem_gb"] is not None:
        print(f"  peak memory  : {r['peak_mem_gb']:.3f} GB")
    print("")
    return r


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse model configuration arguments.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training or inference.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for input data.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--forward_only", action='store_true', help="Benchmark forward pass only.")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--profile", action='store_true', help="Enable profiling.")

    args = parser.parse_args()
    batch_size = args.batch_size
    seq_length = args.seq_length
    num_heads = args.num_heads
    head_dim = args.head_dim
    num_iter = args.num_iter
    forward_only = args.forward_only
    profile = args.profile

    results = []
    try:
        print("Benchmarking flex_attention ...")
        r = run_benchmark(
           batch_size, seq_length, num_heads, head_dim,
           call_flex, forward_only, use_block_mask=True,
           num_iter=num_iter,
           log=True, profile=profile
        )
        results.append(r)
    except Exception as e:
        print("flex_attention benchmark failed:", e)

    try:
        print("Benchmarking scaled_dot_product_attention (SDPA) ...")
        r = run_benchmark(
           batch_size, seq_length, num_heads, head_dim,
           call_sdpa, forward_only,
           num_iter=num_iter,
           log=True, profile=profile
        )
        results.append(r)
    except Exception as e:
        print("SDPA benchmark failed:", e)

    print("\n--- Results ---")
    for r in results:
        print(f"{r['name']}:")
        print(f"  average time     : {r['avg_s'] * 1000:.3f} ms")
        print(f"  TFLOPS   : {r['tflops']}")
        if r["peak_mem_gb"] is not None:
            print(f"  peak memory  : {r['peak_mem_gb']:.3f} GB")
        print("")

