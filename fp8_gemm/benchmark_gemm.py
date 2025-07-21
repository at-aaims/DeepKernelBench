import os
import torch
import torch.nn.functional as F
import argparse


def get_flops(ngpus, m, k, n):
    return ngpus * 2 * m * k * n

# Reference: malfet/scale_mm_example.py
def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()

# https://github.com/pytorch/pytorch/
def scaled_mm_supported_device():
    if torch.cuda.is_available():
        if torch.version.hip:
            return 'gfx94' in torch.cuda.get_device_properties(0).gcnArchName
        else:
            return torch.cuda.get_device_capability() >= (9, 0) or torch.cuda.get_device_capability() == (8, 9)
    return False

def run_benchmark(m, k, n, a_type='float8_e4m3fn',
                  b_type='float8_e4m3fn', o_type='bfloat16',
                  fast_accum=False, f=torch._scaled_mm,
                  warmup_iter=1, num_iter=10,
                  forward_only=True, log=True, profile=False):

    if not scaled_mm_supported_device():
        print("FP8 is only supported on H100+ and sm_89 and MI300+ devices. Skip the benchmark.")
        return

    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    assert m > 0
    assert k > 0
    assert n > 0

    torch_dtypes = {
        'float8_e4m3fn': torch.float8_e4m3fn,
        'float8_e5m2': torch.float8_e5m2,
        'bfloat16': torch.bfloat16,
        'float16' : torch.float16,
        'float32' : torch.float32,
    }

    forward_flops = get_flops(1, m, k, n)
    a = torch.randn(
        m,
        k,
        device=device,
        dtype=torch_dtypes['bfloat16'],
        requires_grad=True,
    )
    b = torch.randn(
        n,
        k,
        device=device,
        dtype=torch_dtypes['bfloat16'],
        requires_grad=True,
    )

    a_f8, a_inv_s = to_float8(a, dtype=torch_dtypes[a_type.strip()])
    b_f8, b_inv_s = to_float8(b, dtype=torch_dtypes[b_type.strip()])
    b_f8 = b_f8.t()
 
    out_dtype = torch_dtypes[o_type.strip()]
    dout = torch.randn(
        m, n, device=device, dtype=out_dtype)

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
        a.grad = None
        b.grad = None
        out = f(
            a_f8,
            b_f8,
            out_dtype=out_dtype,
            scale_a=a_inv_s,
            scale_b=b_inv_s,
            use_fast_accum=bool(fast_accum)
        )
        # RuntimeError: derivative for aten::_scaled_mm is not implemented
        # out.backward(dout)

        cos_sim = F.cosine_similarity(torch.mm(a, b.t()).reshape(-1),
                                      out.reshape(-1), dim=0)
        # Cosine similarity between scaled mm and reference ideally close to 1.0
        print(f'cos_sim {cos_sim.item():.4f}')


    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = f(
                    a_f8,
                    b_f8,
                    out_dtype=out_dtype,
                    scale_a=a_inv_s,
                    scale_b=b_inv_s,
                    use_fast_accum=bool(fast_accum)
                )
                if profile:
                    profiler.step()

    else:
        # RuntimeError: derivative for aten::_scaled_mm is not implemented
        for _ in range(num_iter):
            a.grad = None
            b.grad = None
            _ = f(
                 a_f8,
                 b_f8,
                 out_dtype=out_dtype,
                 scale_a=a_inv_s,
                 scale_b=b_inv_s,
                 use_fast_accum=bool(fast_accum)
            )
            out.backward(dout)
            if profile:
                profiler.step()

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0
    if forward_only:
        TFLOPS = forward_flops/(time/num_iter)/1e12 
    else:
        TFLOPS = 3*forward_flops/(time/num_iter)/1e12 

    if profile:
        profiler.stop()

    print(f"{num_iter / time:.6f} iter/s, {time:.3f} sec, {TFLOPS:.1f} TFLOPS")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse GEMM configuration arguments.")

    parser.add_argument("--m", type=int, default=512, help="Number of rows in Matrix A.")
    parser.add_argument("--k", type=int, default=256, help="Number of columns in Matrix A.")
    parser.add_argument("--n", type=int, default=1024, help="Number of columns in Matrix B.")
    parser.add_argument("--a_type", type=str, default='float8_e4m3fn', help="Precision of Matrix A.")
    parser.add_argument("--b_type", type=str, default='float8_e4m3fn', help="Precision of Matrix B.")
    parser.add_argument("--o_type", type=str, default='bfloat16', help="Precision of Matrix O.")

    # This flag enables CUBLASLT_MATMUL_DESC_FAST_ACCUM here which is defined as: 
    # Flag for managing FP8 fast accumulation mode. When enabled, problem execution might be faster 
    # but at the cost of lower accuracy because intermediate results will not periodically be promoted to a higher precision
    parser.add_argument("--fast_accum", action='store_true', help="Use fast accumulation.")

    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--forward_only", action='store_true', help="Benchmark forward pass only.")
    parser.add_argument("--profile", action='store_true', help="Enable profiling.")
    args = parser.parse_args()

    m = args.m
    k = args.k
    n = args.n
    a_type = args.a_type
    b_type = args.b_type
    o_type = args.o_type
    forward_only = args.forward_only
    fast_accum = args.fast_accum
    profile = args.profile
    num_iter = args.num_iter

    for f in [
        torch._scaled_mm,
    ]:
        torch.cuda.empty_cache()
        #if rank == 0:
        #    print(f"# {f.__name__}")
        run_benchmark(
           m, k, n, a_type, b_type, o_type, fast_accum, f,
           forward_only=forward_only, num_iter=num_iter, log=True, profile=profile
        )
