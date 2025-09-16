import pandas as pd
import time
import torch
from tabulate import tabulate
from triton.testing import do_bench
try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul)
    from vllm.utils.deep_gemm import calc_diff, per_block_cast_to_fp8
    import_vllm = True
except:
    import_vllm = False

print(f"import_vllm : {import_vllm}")

torch.manual_seed(0)
repeats = 200
warmup = 30
timeout = 10

device = 'cuda'
dtype_bf16 = torch.bfloat16

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

    c1 = torch.matmul(a, b1)

    # Block size configuration
    block_size = [128, 128]

    # Pre-quantize A,B
    A_vllm, A_scale_vllm = per_token_group_quant_fp8(a, block_size[1])
    B_vllm, B_scale_vllm = per_block_cast_to_fp8(b2, [128, 128], use_ue8m0=False)

    c2 = w8a8_block_fp8_matmul(A_vllm,
                               B_vllm,
                               A_scale_vllm,
                               B_scale_vllm,
                               block_size,
                               output_dtype=torch.bfloat16)

    vllm_triton_diff = calc_diff(c1, c2)
    print(f"vLLM Triton vs Reference difference: {vllm_triton_diff:.6f}")

    with torch.inference_mode():
        ms = do_bench(lambda: torch.matmul(a, b1), warmup=warmup, rep=repeats)
    gemm_a16w16 = nFLOPS / ms * 1e-9
    time.sleep(timeout)

    if import_vllm:
        ms = do_bench(lambda: w8a8_block_fp8_matmul(A_vllm, B_vllm, A_scale_vllm, B_scale_vllm, block_size, output_dtype=torch.bfloat16), warmup=warmup, rep=repeats)
        gemm_a8w8 = nFLOPS / ms * 1e-9
        time.sleep(timeout)
    else:
        gemm_a8w8 = 0

    # Append Results
    results.append([
        f"({m}, {n}, {k})",
        f"{gemm_a16w16:.1f} TFLOPS",
        f"{gemm_a8w8:.1f} TFLOPS",
    ])

# Print results
headers = [
    "Shape (M, N, K)",
    "bf16 torch.matmul",
    "a8w8 vllm.triton.gemm"
]

table = tabulate(
    results, 
    headers=headers,
    tablefmt="grid"
)
print(f"Benchmark results for Realistic GEMM shapes with {warmup=} and {repeats=}")
print(table)

device_name = torch.cuda.get_device_name(0).replace(' ', '_')
save_file = f"vllm_{device_name}_results.csv"
df = pd.DataFrame.from_records(results, columns=headers)
df.to_csv(save_file)
print(f"Saved results to {save_file}")
