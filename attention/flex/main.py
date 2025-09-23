import time
import torch
import torch.nn.functional as F
import argparse, importlib

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096

try:
    flex_attention = torch.compile(flex_attention)
except Exception as e:
    print("torch.compile(flex_attention) failed or not necessary:", e)

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def call_flex(q, k, v, block_mask):
    return flex_attention(q, k, v, block_mask=block_mask)

def call_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def run_benchmark(batch_size, seqlen, num_heads, head_dim, 
                  fn=call_flex, causal=False, forward_only=False, use_block_mask=False,
                  warmup_iter=10, num_iter=100,
                  log=True, profile=False):
    dtype = torch.bfloat16
    device = torch.device(f"cuda:0")

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(0)
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
    
    block_mask = create_block_mask(
        causal_mask, B=None, H=None, Q_LEN=seqlen, KV_LEN=seqlen, device=device, _compile=True
    )

    torch.cuda.synchronize()

    for i in range(warmup_iter):
        _ = fn(q, k, v, block_mask) if use_block_mask else fn(q, k, v)
    torch.cuda.synchronize()

    begin = torch.cuda.Event(enable_timing=True)
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

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0
    avg = (time/num_iter)/1e12 

    # throughput: tokens/sec = B * L / avg_time
    toks_per_sec = (batch_size * seqlen) / avg if avg > 0 else float("inf")

    peak_mem_gb = None
    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_gb = peak_bytes / (1024**3)

    results = {
        "name": fn.__name__,
        "avg_s": avg,
        "tokens_per_sec": toks_per_sec,
        "peak_mem_gb": peak_mem_gb,
    }
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse model configuration arguments.")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training or inference.")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for input data.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--head_dim", type=int, default=64, help="Dimension of each attention head.")
    parser.add_argument("--causal", action='store_true', help="Enable causal attention masking.")
    parser.add_argument("--forward_only", action='store_true', help="Benchmark forward pass only.")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
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

    results = []
    try:
        print("Benchmarking flex_attention ...")
        r = run_benchmark(
           batch_size, seq_length, num_heads, head_dim,
           call_flex, causal, forward_only,
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
           call_sdpa, causal, forward_only,
           num_iter=num_iter,
           log=True, profile=profile
        )
        results.append(r)
    except Exception as e:
        print("SDPA benchmark failed:", e)

    print("\n--- Results ---")
    for r in results:
        print(f"{r['name']}:")
        print(f"  avg time     : {r['avg_s'] * 1000:.3f} ms")
        print(
            f"  tokens/sec   : {r['tokens_per_sec'] / 1e6:.3f} Mtokens/s ({r['tokens_per_sec']:.0f} toks/s)"
        )
        if r["peak_mem_gb"] is not None:
            print(f"  peak memory  : {r['peak_mem_gb']:.3f} GB")
        print("")

