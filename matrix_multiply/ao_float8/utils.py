# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import collections
import json
import re
import torch
from typing import Optional

import torch.utils.benchmark as benchmark

def get_name_to_shapes_iter(
    shape_gen_name: str,
    M: Optional[int],
    K: Optional[int],
    N: Optional[int],
):
    if shape_gen_name == "llama":
        assert M == K == N == None, (
            f"M, K, N arguments not supported for shape_gen_name {shape_gen_name}"
        )
        bsz, seq_len = 4, 4096
        M = bsz * seq_len
        # LLaMa 2 70B single-node weight shapes
        # assumes fused attn.wqkv and ffn.w13
        # source: https://fburl.com/gsheet/g8onr7rh
        name_to_shapes_70b = {
            "attn.wqkv": (M, 8192, 1280),
            "attn.w0": (M, 1024, 8192),
            "ffn.w13": (M, 8192, 7168),
            "ffn.w2": (M, 3584, 8192),
        }
        return name_to_shapes_70b.items()

    elif shape_gen_name == "forgeL":
        assert M == K == N == None, (
            f"M, K, N arguments not supported for shape_gen_name {shape_gen_name}"
        )
        bsz, seq_len = 16, 2048
        M = bsz * seq_len
        h, t, v = 6144, 2, 52000
        
        name_to_shapes = {
            "QKV_transform": (M, h, 3 * h // t),
            "Linear_project": (M, h//t, h),
            "MLP1": (M, h, 4 * h // t),
            "MLP2": (M, 4 * h // t, h),
            "Linear_output": (M, v, h),
        }
        return name_to_shapes.items()

    elif shape_gen_name == "pow2":
        assert M == K == N == None, (
            f"M, K, N arguments not supported for shape_gen_name {shape_gen_name}"
        )
        name_to_shapes = {}
        min_power_of_2 = 10  # 1024
        max_power_of_2 = 14  # 16,384
        for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
            val = 2**power_of_2
            name_to_shapes[idx] = val, val, val
        return name_to_shapes.items()

    elif shape_gen_name == "pow2_extended":
        assert M == K == N == None, (
            f"M, K, N arguments not supported for shape_gen_name {shape_gen_name}"
        )
        name_to_shapes = {}
        min_power_of_2 = 10  # 1024
        max_power_of_2 = 14  # 16,384
        for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
            val1 = 2**power_of_2
            name_to_shapes[idx * 2] = val1, val1, val1
            val2 = 2**power_of_2 + 2 ** (power_of_2 - 1)
            name_to_shapes[idx * 2 + 1] = val2, val2, val2
        return name_to_shapes.items()

    elif shape_gen_name == "sweep":
        assert M == K == N == None, (
            f"M, K, N arguments not supported for shape_gen_name {shape_gen_name}"
        )
        name_to_shapes = {}
        min_p2 = 8  # 256
        max_p2 = 15  # 32,768
        counter = 0
        for M_p2 in range(min_p2, max_p2 + 1):
            M = 2**M_p2
            for K_p2 in range(min_p2, max_p2 + 1):
                K = 2**K_p2
                for N_p2 in range(min_p2, max_p2 + 1):
                    N = 2**N_p2
                    name_to_shapes[counter] = M, K, N
                    counter += 1
        return name_to_shapes.items()

    elif shape_gen_name == "custom":
        assert M is not None and K is not None and N is not None, (
            "M, K, N must be specified for custom shape_gen"
        )
        name_to_shapes = {
            1: (M, K, N),
        }
        return name_to_shapes.items()

    raise AssertionError(f"unknown shape_gen_name {shape_gen_name}")


def get_name_to_moe_shapes_iter(
    shape_gen_name: str,
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    E: Optional[int] = None,
):
    M = 16640 if M is None else M
    if shape_gen_name == "llama4_17bx16e":
        # num_experts=16, dim=5120
        names_to_shapes = {
            # M, K, N, E
            "moe.experts.w1": (M, 5120, 8192, 16),
            "moe.experts.w2": (M, 8192, 5120, 16),
        }
        return names_to_shapes.items()
    elif shape_gen_name == "llama4_17bx128e":
        # num_experts=128, dim=5120
        names_to_shapes = {
            # M, K, N, E
            "moe.experts.w1": (M, 5120, 4 * 5120, 128),
            "moe.experts.w2": (M, 4 * 5120, 5120, 128),
        }
        return names_to_shapes.items()
    elif shape_gen_name == "custom":
        assert M is not None and K is not None and N is not None and E is not None, (
            "M, K, N, E must be specified for custom shape_gen"
        )
        name_to_shapes = {
            1: (M, K, N, E),
        }
        return name_to_shapes.items()

    raise AssertionError(f"unknown shape_gen_name {shape_gen_name}")


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(30):
        f(*args, **kwargs)

    torch.cuda.synchronize()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()
    with torch.no_grad():
        for _ in range(100):
            f(*args, **kwargs)

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize()
    time = begin.elapsed_time(end) / 1000.0
    return time / 100
    
    #t0 = benchmark.Timer(
    #    stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    #)
    #measurement = t0.blocked_autorange(min_run_time=1)
    #return measurement.mean


def do_benchmarks(
    tops,
    peak_tops,
    f,
    *args,
    **kwargs,
):
    # e2e time including kernel launch overhead
    time_sec = benchmark_fn_in_sec(f, *args, **kwargs)
    tops_sec = tops / time_sec
    pct_top_peak = tops_sec / peak_tops
    return time_sec, tops_sec, pct_top_peak


gpu_name_to_specs = {
    "NVIDIA H200": {
        "fp32_peak_tops": 67e12,
        "fp16_peak_tops": 989e12,
        "bf16_peak_tops": 989e12,
        "int8_peak_tops": 1979e12,
        "fp8_peak_tops" : 1979e12,
    },
    "NVIDIA H200 NVL": {
        "fp32_peak_tops": 60e12,
        "fp16_peak_tops": 835e12,
        "bf16_peak_tops": 835e12,
        "int8_peak_tops": 1670e12,
        "fp8_peak_tops" : 1670e12,
    },
    "NVIDIA H100 80GB HBM3" : {
        # https://www.nvidia.com/en-us/data-center/h100/, divide by 2 because no sparsity
         # H100 SXM specs: bottom of https://www.nvidia.com/en-us/data-center/h100/
        "fp32_peak_tops": 67e12,
        "fp16_peak_tops": 989e12,
        "bf16_peak_tops": 989e12,
        "int8_peak_tops": 1979e12,
        "fp8_peak_tops" : 1979e12,
    },
    "NVIDIA H100 NVL": {
        "fp32_peak_tops": 60e12,
        "fp16_peak_tops": 835e12,
        "bf16_peak_tops": 835e12,
        "int8_peak_tops": 1670e12,
        "fp8_peak_tops" : 1670e12,
    },
    "AMD Instinct MI300X": {
        # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf, page 1,
        "fp32_peak_tops": 163e12,
        "fp16_peak_tops": 1307e12,
        "bf16_peak_tops": 1307e12,
        "int8_peak_tops": 2614e12,
        "fp8_peak_tops" : 2614e12,
    },
    "AMD Instinct MI300A": {
        "fp32_peak_tops": 122e12,
        "fp16_peak_tops": 980e12,
        "bf16_peak_tops": 980e12,
        "int8_peak_tops": 1961e12,
        "fp8_peak_tops" : 1961e12,
    },
}

def get_peak_tops_from_spec (gpu_name):
    spec = gpu_name_to_specs[gpu_name]
    dtype_to_peak_tops = {
        torch.float32:       spec["fp32_peak_tops"],
        torch.float16:       spec["bf16_peak_tops"],
        torch.bfloat16:      spec["bf16_peak_tops"],
        torch.int8:          spec["int8_peak_tops"],
        torch.float8_e4m3fn: spec["fp8_peak_tops"],
        torch.float8_e5m2:   spec["fp8_peak_tops"],
    }
    return dtype_to_peak_tops

