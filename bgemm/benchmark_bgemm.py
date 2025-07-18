import os
import torch
import argparse

def get_flops(ngpus, batch_size, m, k, n):
    return ngpus * batch_size * 2 * m * k * n

# https://docs.pytorch.org/docs/stable/generated/torch.bmm.html#torch.bmm
def run_benchmark(batch_size, m, k, n, a_type='bfloat16',
                  b_type='bfloat16', o_type='bfloat16',
                  f=torch.bmm, warmup_iter=1, num_iter=10, 
                  forward_only=True, log=True, profile=False):
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    assert batch_size > 0
    assert m > 0
    assert k > 0
    assert n > 0

    torch_dtypes = {
        'bfloat16': torch.bfloat16,
        'float16' : torch.float16,
        'float32' : torch.float32,
    }

    forward_flops = get_flops(1, batch_size, m, k, n)
    a = torch.randn(
        batch_size,
        m,
        k,
        device=device,
        dtype=torch_dtypes[a_type.strip()],
        requires_grad=True,
    )

    b = torch.randn(
        batch_size,
        k,
        n,
        device=device,
        dtype=torch_dtypes[b_type.strip()],
        requires_grad=True,
    )
 
    dout = torch.randn(
        batch_size, m, n, device=device, dtype=torch_dtypes[o_type.strip()]
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
        a.grad = None
        b.grad = None
        out = f(
            a,
            b,
        )
        out.backward(dout)


    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = f(
                    a,
                    b
                )
                if profile:
                    profiler.step()

    else:
        for _ in range(num_iter):
            a.grad = None
            b.grad = None
            out = f(
                a,
                b,
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

    parser = argparse.ArgumentParser(description="Parse batched GEMM configuration arguments.")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training or inference.")
    parser.add_argument("--m", type=int, default=1024, help="Number of rows of Matrix A.")
    parser.add_argument("--k", type=int, default=1024, help="Number of columns of Matrix A.")
    parser.add_argument("--n", type=int, default=1024, help="Number of columns of Matrix B.")
    parser.add_argument("--a_type", type=str, default='bfloat16', help="Precision of Matrix A.")
    parser.add_argument("--b_type", type=str, default='bfloat16', help="Precision of Matrix B.")
    parser.add_argument("--o_type", type=str, default='bfloat16', help="Precision of Matrix O.")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--forward_only", action='store_true', help="Benchmark forward pass only.")
    parser.add_argument("--profile", action='store_true', help="Enable profiling.")
    args = parser.parse_args()

    batch_size = args.batch_size
    m = args.m
    k = args.k
    n = args.n
    a_type = args.a_type
    b_type = args.b_type
    o_type = args.o_type
    forward_only = args.forward_only
    profile = args.profile
    num_iter = args.num_iter

    for f in [
        torch.bmm,
    ]:
        torch.cuda.empty_cache()
        #if rank == 0:
        #    print(f"# {f.__name__}")
        run_benchmark(
           batch_size, m, k, n, a_type, b_type, o_type, f,
           forward_only=forward_only, num_iter=num_iter, log=True, profile=profile
        )
