import torch.nn.functional as F
import pandas as pd
import time
import torch
from tabulate import tabulate
from triton.testing import do_bench
from typing import Union

try:
    from aiter.ops.triton.gemm_a8w8 import gemm_a8w8
    from aiter.ops.triton.utils.arch_info import get_fp8_dtypes
    import_aiter = True
except:
    import_aiter = False

print(f"import_aiter : {import_aiter}")

def generate_gemm_a8w8_inputs(
    M: int,
    N: int,
    K: int,
    in_dtype: Union[torch.dtype, str],
    out_dtype: Union[torch.dtype, str],
    layout: str = "TN",
    output=False,
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    """
    if layout[0] == "T":
        # T (transposed) in Fortran notation equals row-major
        x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    else:
        x = torch.randn((K, M), dtype=torch.float32, device="cuda").T

    if layout[1] == "N":
        weight = torch.randn((N, K), dtype=torch.float32, device="cuda")
    else:
        weight = torch.randn((K, N), dtype=torch.float32, device="cuda").T

    max_x = x.abs().float().amax(dim=1, keepdim=True)

    e5m2_type, e4m3_type = get_fp8_dtypes()
    dtype_max = {
      dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
      for dtype in [
          e5m2_type,
          e4m3_type,
          torch.int8,
      ]
    }
    x_scale = max_x / dtype_max[in_dtype]
    x = x / x_scale
    x = x.to(in_dtype)

    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / dtype_max[in_dtype]
    weight = weight / w_scale.T
    weight = weight.to(in_dtype)

    bias = torch.rand([1, N], dtype=torch.float32, device="cuda") * 10

    y = None
    if output:
        y = torch.empty((M, N), dtype=out_dtype, device="cuda")

    return x, weight, x_scale, w_scale, bias, y

def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)

def run_triton(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16, y=None):
    return gemm_a8w8(x, weight, x_scale, w_scale, bias, dtype, y)

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

    x, weight, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
        m, n, k, dtype_fp8_e4m3, dtype_bf16, layout="TN", output=True
    )

    a = run_torch(x, weight, x_scale, w_scale, bias, dtype_bf16)
    b = run_triton(x, weight, x_scale, w_scale, bias, dtype_bf16, y)

    try:
        torch.testing.assert_close(a, b, atol=0.02, rtol=1e-2)
    except AssertionError as error:
        print(error)

    if import_aiter:
        ms_fp8 = do_bench(lambda: gemm_a8w8(x, weight, x_scale, w_scale, bias, dtype_bf16, y), warmup=warmup, rep=repeats)
        gemm_a8w8_tflops = nFLOPS / ms_fp8 * 1e-9
    else:
        gemm_a8w8_tflops = 0


    # Append Results
    results.append([
        f"({m}, {n}, {k})",
        f"{gemm_a8w8_tflops:.1f} TFLOPS",
    ])

# Print results
headers = [
    "Shape (M, N, K)",
    "bf8 aiter.gemm_a8w8"
]

table = tabulate(
    results, 
    headers=headers,
    tablefmt="grid"
)
print(f"Benchmark results for Realistic GEMM shapes with {warmup=} and {repeats=}")
print(table)

device_name = torch.cuda.get_device_name(0).replace(' ', '_')
save_file = f"aiter_a8w8_{device_name}_results.csv"
df = pd.DataFrame.from_records(results, columns=headers)
df.to_csv(save_file)
print(f"Saved results to {save_file}")
