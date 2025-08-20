import pandas as pd
import time
import torch
from tabulate import tabulate
from triton.testing import do_bench
try:
    import tritonblas
    import_tb = True
except:
    import_tb = False

print(f"import_tb : {import_tb}")

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
    b = torch.randn(n, k, device=device, dtype=dtype_bf16).transpose(-1, -2)
    c = torch.zeros(m, n, device=device, dtype=dtype_bf16)

    torch_c = torch.matmul(a, b)
    #print("verify tritonblas.matmul(a, b, c)")
    tritonblas.matmul(a, b, c)
    torch.testing.assert_close(torch_c, c, atol=1, rtol=1)

    #print("verify tritonblas.matmul_lt(a, b, c)")
    selector = tritonblas.MatmulHeuristicResult(m, n, k, a.dtype, b.dtype, c.dtype)
    tritonblas.matmul_lt(a, b, c, selector)
    torch.testing.assert_close(torch_c, c, atol=1, rtol=1)

    with torch.inference_mode():
        ms_bf16 = do_bench(lambda: torch.matmul(a, b, out=torch_c), warmup=warmup, rep=repeats)
    matmul_tflops_bf16 = nFLOPS / ms_bf16 * 1e-9
    time.sleep(timeout)

    if import_tb:
        ms_bf16 = do_bench(lambda: tritonblas.matmul(a, b, c), warmup=warmup, rep=repeats)
        tb_matmul_tflops_bf16 = nFLOPS / ms_bf16 * 1e-9
        time.sleep(timeout)

        ms_bf16 = do_bench(lambda: tritonblas.matmul_lt(a, b, c, selector), warmup=warmup, rep=repeats)
        tb_matmul_lt_tflops_bf16 = nFLOPS / ms_bf16 * 1e-9
        time.sleep(timeout)

    else:
        tb_matmul_tflops_bf16 = 0
        tb_matmul_lt_tflops_bf16 = 0


    print(f"({m}, {n}, {k})",
          f"{matmul_tflops_bf16:.1f} TFLOPS",
          f"{tb_matmul_tflops_bf16:.1f} TFLOPS",
          f"{tb_matmul_lt_tflops_bf16:.1f} TFLOPS")

    # Append Results
    results.append([
        f"({m}, {n}, {k})",
        f"{matmul_tflops_bf16:.1f} TFLOPS",
        f"{tb_matmul_tflops_bf16:.1f} TFLOPS",
        f"{tb_matmul_lt_tflops_bf16:.1f} TFLOPS"
    ])

# Print results
headers = [
    "Shape (M, N, K)",
    "bf16 torch.matmul",
    "bf16 tritonblas.matmul",
    "bf16 tritonblas.matmul_lt",
]

table = tabulate(
    results, 
    headers=headers,
    tablefmt="grid"
)
print(f"Benchmark results for Realistic GEMM shapes with {warmup=} and {repeats=}")
print(table)

device_name = torch.cuda.get_device_name(0).replace(' ', '_')
save_file = f"tritonBLAS_{device_name}_results.csv"
df = pd.DataFrame.from_records(results, columns=headers)
df.to_csv(save_file)
print(f"Saved results to {save_file}")
