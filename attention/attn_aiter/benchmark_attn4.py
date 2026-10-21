import argparse
import os
import sys
import torch

try:
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4
    try:
        from aiter.ops.mha_v4 import native_fp8_format
    except ImportError:
        native_fp8_format = None
except ImportError as e:
    print(f"Error: Required library is not installed or available. {e}")
    print("Please install aiter with MHA v4 to run this script.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# End-to-end mha_v4 call. Quantized recipes include Q/K/V quantization in the timed region.
_RECIPES = ("bf16", "fp8", "mxfp4")
_TOLERANCE = {
    "bf16": (1e-2, 1e-2),
    "fp8": (1e-1, 1e-1),
    "mxfp4": (1e-1, 1e-1),
}


def get_flops(ngpus, batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    s = ngpus * seqlen
    f = 4 * batch * s**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def _fp8_format():
    """FP8 encoding native to the current GPU. gfx942 uses E4M3 FNUZ."""
    if native_fp8_format is not None:
        return native_fp8_format()
    gfx = ""
    try:
        from aiter.jit.utils.chip_info import get_gfx
        gfx = get_gfx()
    except Exception:
        gfx = ""
    if gfx == "gfx942" and hasattr(AttentionFormat, "FP8_E4M3_FNUZ"):
        return AttentionFormat.FP8_E4M3_FNUZ
    return AttentionFormat.FP8


def _formats(recipe):
    recipe = str(recipe).strip().lower()
    if recipe == "bf16":
        fmt = AttentionFormat.BF16
        return recipe, fmt, fmt, fmt
    if recipe == "fp8":
        fmt = _fp8_format()
        return recipe, fmt, fmt, fmt
    if recipe == "mxfp4":
        fmt = AttentionFormat.MXFP4
        return recipe, fmt, fmt, fmt
    raise ValueError(
        f"unknown MHA v4 recipe {recipe!r}; expected one of {', '.join(_RECIPES)}"
    )


def mha_v4_recipe(q, k, v, q_format, k_format, v_format):
    """Dense MHA v4. Inputs stay BF16; the format selects the quantized recipe."""
    return mha_v4(
        q,
        k,
        v,
        q_format,
        k_format,
        v_format,
        softmax_scale=q.shape[-1] ** -0.5,
    )


def run_benchmark(batch_size, seqlen, num_heads, head_dim,
                  causal=False, forward_only=True,
                  recipe="bf16",
                  warmup_iter=1000, num_iter=1000,
                  log=True, profile=False):
    """Benchmark aiter MHA v4.

    Contiguous BF16 BSHD, head_dim 128, non-causal, forward-only.
    recipe is bf16, fp8, or mxfp4. fp8 and mxfp4 time quantization plus the kernel.
    Assembly objects ship for gfx942 (MI300) and gfx950 (MI350/MI355).
    """
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    causal = bool(causal)
    forward_only = bool(forward_only)
    recipe, q_format, k_format, v_format = _formats(recipe)

    if head_dim != 128:
        raise ValueError(
            f"aiter MHA v4 supports head_dim=128 only, got {head_dim}"
        )
    if causal:
        raise ValueError("aiter MHA v4 does not support causal masking")
    if not forward_only:
        raise ValueError("aiter MHA v4 is forward-only (no backward)")

    scope = "kernel" if recipe == "bf16" else "end-to-end, includes Q/K/V quantization"
    print(
        f"backend: aiter.ops.mha_v4 {recipe} "
        f"Q/K={q_format.name} V={v_format.name} ({scope}, non-causal, forward)"
    )

    def launch(q, k, v):
        return mha_v4_recipe(q, k, v, q_format, k_format, v_format)

    torch.manual_seed(20)
    q = torch.empty(
        (batch_size, seqlen, num_heads, head_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    k = torch.empty(
        (batch_size, seqlen, num_heads, head_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)
    v = torch.empty(
        (batch_size, seqlen, num_heads, head_dim), dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.5)

    try:
        with torch.no_grad():
            ref = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2).float(),
                k.transpose(1, 2).float(),
                v.transpose(1, 2).float(),
            ).transpose(1, 2)
            out = launch(q, k, v)
        max_abs = (out.float() - ref).abs().max().item()
        cosine = torch.nn.functional.cosine_similarity(
            out.float().flatten(), ref.flatten(), dim=0
        ).item()
        print(f"max abs {max_abs:.4e}, cosine {cosine:.6f}")
        atol, rtol = _TOLERANCE[recipe]
        torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)
    except Exception as e:
        print('--------------------------------------------------------------------------------')
        print('Exceptions raised during correctness check:')
        print(e)
        print('--------------------------------------------------------------------------------')

    torch.cuda.empty_cache()
    q = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)

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
                os.path.join("./benchmark/logs/mha_v4", f"rank_0_{recipe}")
            ),
        )

    try:
        with torch.no_grad():
            for _ in range(warmup_iter):
                launch(q, k, v)
            torch.cuda.synchronize(device=device)

            if profile:
                profiler.start()

            begin = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(num_iter):
                launch(q, k, v)
                if profile:
                    profiler.step()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize(device=device)

        if profile:
            profiler.stop()

        time = begin.elapsed_time(end) / 1000.0
        flops = get_flops(1, batch_size, seqlen, num_heads, head_dim, False, "fwd")
        tflops = flops / (time / num_iter) / 1e12
        if log:
            print(f"{recipe}: {num_iter / time:.6f} iter/s, {time:.3f} sec, {tflops:.1f} TFLOPS")
    except Exception as e:
        print('--------------------------------------------------------------------------------')
        print('Exceptions raised during benchmark execution:')
        print(e)
        print('--------------------------------------------------------------------------------')
        tflops = 0

    return tflops


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark aiter MHA v4 (AMD FA4).")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension. MHA v4 requires 128.")
    parser.add_argument("--causal", action="store_true", help="Rejected: MHA v4 is non-causal.")
    parser.add_argument("--forward_only", action="store_true", default=True, help="Forward pass only. Required.")
    parser.add_argument("--recipe", default="bf16", choices=_RECIPES, help="bf16, fp8, or mxfp4.")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--profile", action="store_true", help="Enable profiling.")

    args = parser.parse_args()

    torch.cuda.empty_cache()
    run_benchmark(
        args.batch_size, args.seq_length, args.num_heads, args.head_dim,
        args.causal, args.forward_only,
        recipe=args.recipe,
        num_iter=args.num_iter,
        log=True, profile=args.profile,
    )
