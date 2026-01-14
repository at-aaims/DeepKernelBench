"""
Helion Matmul Kernel Example with autotuning
"""

# %%
from __future__ import annotations

import helion
from helion._testing import DEVICE
import helion.language as hl

import torch
from torch import Tensor

from triton.testing import do_bench

import pandas as pd
from tabulate import tabulate
import time


# disable autotuning with autotune_effort="none"
@helion.kernel()
def helion_matmul(
    x: Tensor,
    y: Tensor,
    out: Tensor
) -> None:
    """
    Performs matrix multiplication of x and y with an optional epilogue function.
    Args:
        x (Tensor): Left matrix of shape [m, k].
        y (Tensor): Right matrix of shape [k, n].
    Returns:
        Tensor: Resulting matrix of shape [m, n].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc


# autotuning for each input
def autotune(x, y, out, m, k, n):
    """
    Runs autotuning on the matmul kernel with a ReLU epilogue and saves the best config.
    Args:
        m (int): Number of rows in matrix x.
        k (int): Number of columns in matrix x and rows in matrix y.
        n (int): Number of columns in matrix y.
    """
    args = (x, y, out)
    best_config = helion_matmul.autotune(args)
    best_config.save(f"configs/matmul_m{m}_k{k}_n{n}.json")


if __name__ == "__main__":

    torch.manual_seed(0)
    repeats = 200
    warmup = 30
    timeout = 10
    
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
        a = torch.randn(m, k, device=DEVICE, dtype=dtype_bf16)
        b = torch.randn(k, n, device=DEVICE, dtype=dtype_bf16)
    
        c1 = torch.zeros(m, n, device=DEVICE, dtype=dtype_bf16)
        c2 = torch.zeros(m, n, device=DEVICE, dtype=dtype_bf16)
    
        torch.matmul(a, b, out=c1)
        autotune(a, b, c2, m, k, n)
        helion_matmul(a, b, c2)
    
        try:
            torch.testing.assert_close(c1, c2, atol=0.02, rtol=1e-2)
        except AssertionError as error:
            print(error)
    
        with torch.inference_mode():
            ms_bf16 = do_bench(lambda: torch.matmul(a, b, out=c1), warmup=warmup, rep=repeats)
        torch_matmul_tflops_bf16 = nFLOPS / ms_bf16 * 1e-9

        time.sleep(timeout)
    
        ms_bf16 = do_bench(lambda: helion_matmul(a, b, c2), warmup=warmup, rep=repeats)
        helion_matmul_tflops_bf16 = nFLOPS / ms_bf16 * 1e-9
    
        # Append Results
        results.append([
            f"({m}, {n}, {k})",
            f"{torch_matmul_tflops_bf16:.1f} TFLOPS",
            f"{helion_matmul_tflops_bf16:.1f} TFLOPS",
        ])
    
    # Print results
    headers = [
        "Shape (M, N, K)",
        "bf16 torch.matmul",
        "bf16 helion.matmul"
    ]
    
    table = tabulate(
        results,
        headers=headers,
        tablefmt="grid"
    )
    print(f"Benchmark results for Realistic GEMM shapes with {warmup=} and {repeats=}")
    print(table)
    
    if DEVICE.type == 'xpu':
        device_name = torch.xpu.get_device_name(0).replace(' ', '_')
    else:
        device_name = torch.cuda.get_device_name(0).replace(' ', '_')

    save_file = f"helion_matmul_{device_name}_results.csv"
    df = pd.DataFrame.from_records(results, columns=headers)
    df.to_csv(save_file)
    print(f"Saved results to {save_file}")
