import pandas as pd
import time
import torch
from tabulate import tabulate
from triton.testing import do_bench
try:
    from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
    import_aiter = True
except:
    import_aiter = False

print(f"import_aiter : {import_aiter}")

torch.manual_seed(0)
repeats = 200
warmup = 30
timeout = 10
is_nvidia = "nvidia" in torch.cuda.get_device_name(0).lower()

device = 'cuda'
dtype_bf16 = torch.bfloat16
dtype_fp8_e5m2 = torch.float8_e5m2
dtype_fp8_e4m3 = torch.float8_e4m3fn if is_nvidia else torch.float8_e4m3fnuz

# GEMM Shapes
shapes = [
    (16384, 8192, 1280),
    (16384, 1024, 8192),
    (16384, 8192, 7168),
    (16384, 3584, 8192),
    (8192, 8192, 8192)
]

results = []

for (m, n, k) in shapes:
    # FLOPS
    nFLOPS = 2 * m * n * k

    # Matmul benchmark in bf16
    a = torch.randn(m, k, device=device, dtype=dtype_bf16)
    b1 = torch.randn(k, n, device=device, dtype=dtype_bf16)
    b2 = b1.transpose(-1, -2)

    c1 = torch.zeros(m, n, device=device, dtype=dtype_bf16)
    c2 = torch.zeros(m, n, device=device, dtype=dtype_bf16)

    torch.matmul(a, b1, out=c1)
    # x.shape[1] == w.shape[1]
    gemm_a16w16(a, b2, dtype_bf16, c2, activation=None)

    try:
        torch.testing.assert_close(c1, c2)
    except AssertionError as error:
        print(error)

    with torch.inference_mode():
        ms_bf16 = do_bench(lambda: torch.matmul(a, b1, out=c1), warmup=warmup, rep=repeats)
    matmul_tflops_bf16 = nFLOPS / ms_bf16 * 1e-9
    time.sleep(timeout)

    if import_aiter:
        ms_bf16 = do_bench(lambda: gemm_a16w16(a, b2, dtype_bf16, c2, activation=None), warmup=warmup, rep=repeats)
        gemm_a16w16_tflops_bf16 = nFLOPS / ms_bf16 * 1e-9
        time.sleep(timeout)
    else:
        gemm_a16w16_tflops_bf16 = 0


    # Append Results
    results.append([
        f"({m}, {n}, {k})",
        f"{matmul_tflops_bf16:.1f} TFLOPS",
        f"{gemm_a16w16_tflops_bf16:.1f} TFLOPS",
    ])

# Print results
headers = [
    "Shape (M, N, K)",
    "bf16 torch.matmul",
    "bf16 aiter.gemm_a16w16"
]

table = tabulate(
    results, 
    headers=headers,
    tablefmt="grid"
)
print(f"Benchmark results for Realistic GEMM shapes with {warmup=} and {repeats=}")
print(table)

device_name = torch.cuda.get_device_name(0).replace(' ', '_')
save_file = f"aiter_a16w16_{device_name}_results.csv"
df = pd.DataFrame.from_records(results, columns=headers)
df.to_csv(save_file)
print(f"Saved results to {save_file}")
