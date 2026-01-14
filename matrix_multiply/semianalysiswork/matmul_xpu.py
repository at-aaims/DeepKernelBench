import pandas as pd
import time
import torch
from tabulate import tabulate
from triton.testing import do_bench

torch.manual_seed(0)
repeats = 200
warmup = 30
timeout = 10

device = 'xpu'
dtype_bf16 = torch.bfloat16
dtype_fp8_e5m2 = torch.float8_e5m2
dtype_fp8_e4m3 = torch.float8_e4m3fn

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
    with torch.inference_mode():
        ms_bf16 = do_bench(lambda: torch.matmul(a, b), warmup=warmup, rep=repeats)
    tflops_bf16 = nFLOPS / ms_bf16 * 1e-9
    time.sleep(timeout)

    try:
        # FP8 e5m2 torch._scaled_mm (A: e5m2, B: e4m3fn)
        a_fp8_e5m2 = torch.randn(m, k, device=device).to(dtype_fp8_e5m2)
        b_fp8_e4m3 = torch.randn(n, k, device=device).to(dtype_fp8_e4m3).transpose(-1, -2)
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        with torch.inference_mode():
            ms_fp8_scaled_mm_e5m2 = do_bench(lambda: torch._scaled_mm(a_fp8_e5m2, b_fp8_e4m3, scale_a, scale_b), warmup=warmup, rep=repeats)
        tflops_fp8_scaled_mm_e5m2 = nFLOPS / ms_fp8_scaled_mm_e5m2 * 1e-9
        time.sleep(timeout)
    except:
        tflops_fp8_scaled_mm_e5m2 = 0.00

    # FP8 e4m3 torch._scaled_mm
    a_fp8_e4m3 = torch.randn(m, k, device=device).to(dtype_fp8_e4m3)
    b_fp8_e4m3 = torch.randn(n, k, device=device).to(dtype_fp8_e4m3).transpose(-1, -2)
    scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
    scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
    with torch.inference_mode():
        try:
            ms_fp8_scaled_mm_e4m3 = do_bench(lambda: torch._scaled_mm(a_fp8_e4m3, b_fp8_e4m3), warmup=warmup, rep=repeats)
        except:
            ms_fp8_scaled_mm_e4m3 = do_bench(lambda: torch._scaled_mm(a_fp8_e4m3, b_fp8_e4m3, scale_a, scale_b), warmup=warmup, rep=repeats)
    tflops_fp8_scaled_mm_e4m3 = nFLOPS / ms_fp8_scaled_mm_e4m3 * 1e-9
    time.sleep(timeout)

    # Append Results
    results.append([
        f"({m}, {n}, {k})",
        f"{tflops_bf16:.1f} TFLOPS",
        f"{tflops_fp8_scaled_mm_e5m2:.1f} TFLOPS",
        f"{tflops_fp8_scaled_mm_e4m3:.1f} TFLOPS"
    ])

# Print results
headers = [
    "Shape (M, N, K)",
    "bf16 torch.matmul",
    "FP8 torch._scaled_mm (e5m2/e4m3fn)",
    "FP8 torch._scaled_mm (e4m3fn)"
]

table = tabulate(
    results, 
    headers=headers,
    tablefmt="grid"
)
print(f"Benchmark results for Realistic GEMM shapes with {warmup=} and {repeats=}")
print(table)

device_name = torch.xpu.get_device_name(0).replace(' ', '_')
save_file = f"semianalysis_{device_name}_results.csv"
df = pd.DataFrame.from_records(results, columns=headers)
df.to_csv(save_file)
print(f"Saved results to {save_file}")
