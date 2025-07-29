import time
import torch
from tabulate import tabulate
from triton.testing import do_bench
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    import_te = True
except:
    import_te = False

print(f"import_te : {import_te}")

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

# FP8 Recipe (for scaling)
if import_te:
    fp8_format = recipe.Format.HYBRID
    fp8_recipe = recipe.DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")

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

    # TE Linear (with FP8 autocast) benchmark
    if import_te:
        input_tensor = torch.randn(m, k, device=device)
        linear_layer = te.Linear(k, n, bias=False).to(device)
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            ms_te_linear = do_bench(lambda: linear_layer(input_tensor), warmup=warmup, rep=repeats)
        tflops_te_linear = nFLOPS / ms_te_linear * 1e-9
        time.sleep(timeout)
    else:
        tflops_te_linear = 0.0

    if is_nvidia:
        # FP8 e5m2 torch._scaled_mm (A: e5m2, B: e4m3fn)
        a_fp8_e5m2 = torch.randn(m, k, device=device).to(dtype_fp8_e5m2)
        b_fp8_e4m3 = torch.randn(n, k, device=device).to(dtype_fp8_e4m3).transpose(-1, -2)
        scale_a = torch.tensor(1.0, device=device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=device, dtype=torch.float32)
        with torch.inference_mode():
            ms_fp8_scaled_mm_e5m2 = do_bench(lambda: torch._scaled_mm(a_fp8_e5m2, b_fp8_e4m3, scale_a, scale_b), warmup=warmup, rep=repeats)
        tflops_fp8_scaled_mm_e5m2 = nFLOPS / ms_fp8_scaled_mm_e5m2 * 1e-9
        time.sleep(timeout)
    else:
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
        f"{tflops_te_linear:.1f} TFLOPS",
        f"{tflops_fp8_scaled_mm_e5m2:.1f} TFLOPS",
        f"{tflops_fp8_scaled_mm_e4m3:.1f} TFLOPS"
    ])

# Print results
headers = [
    "Shape (M, N, K)",
    "bf16 torch.matmul",
    "FP8 TE.Linear (autocast, bias=False)",
    "FP8 torch._scaled_mm (e5m2/e4m3fn)",
    "FP8 torch._scaled_mm (e4m3)"
]

table = tabulate(
    results, 
    headers=headers,
    tablefmt="grid"
)
print(f"Benchmark results for Realistic GEMM shapes with {warmup=} and {repeats=}")
print(table)

