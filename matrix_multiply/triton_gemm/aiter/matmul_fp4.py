# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import argparse
import pandas as pd

try:
    import aiter
    from aiter.test_common import checkAllclose, benchmark, perftest, run_perftest
    from aiter import dtypes
    from aiter.utility import fp4_utils
    from aiter.ops.shuffle import shuffle_weight
    import_aiter = True
except:
    import_aiter = False

print(f"import_aiter : {import_aiter}")

if not import_aiter:
    print("aiter is not imported successfully. Skip the benchmark")
    exit(1)
else:
    from aiter.jit.utils.chip_info import get_gfx
    if get_gfx() not in ["gfx950"]:
        print("FP4 is only supported on gfx950+ devices. Skip the benchmark.")
        exit(1)
    else:
        l_dtype = ["bf16"]
        l_mnk = [
            # pure_compute
            (256, 2048, 8192),
            (2048, 8192, 8192),
            (16384, 16384, 16384),
            (32768, 106496, 16384),
            (32768, 16384, 53248),
            (32768, 18432, 16384),
            (32768, 16384, 16384),
            (128, 106496, 16384),
            (128, 16384, 53248),
            (128, 18432, 16384),
            (128, 16384, 16384),
            (64, 106496, 16384),
            (64, 16384, 53248),
            (64, 18432, 16384),
            (64, 16384, 16384),
            (64, 106496, 16384),
            (32, 106496, 16384),
            (32, 16384, 53248),
            (32, 18432, 16384),
            (32, 16384, 16384),
            # qkv_proj
            (1, 1280, 8192),
            (64, 1280, 8192),
            (127, 1280, 8192),
            (129, 1280, 8192),
            (65, 1280, 8192),
            (32, 1280, 8192),
            (64, 1280, 8192),
            (128, 1280, 8192),
            (192, 1280, 8192),
            (256, 1280, 8192),
            (320, 1280, 8192),
            (512, 1280, 8192),
            (1024, 1280, 8192),
            (2048, 1280, 8192),
            (4096, 1280, 8192),
            (8192, 1280, 8192),
            # attn_out
            (1, 8192, 1024),
            (32, 8192, 1024),
            (64, 8192, 1024),
            (128, 8192, 1024),
            (192, 8192, 1024),
            (256, 8192, 1024),
            (320, 8192, 1024),
            (512, 8192, 1024),
            (1024, 8192, 1024),
            (2048, 8192, 1024),
            (4096, 8192, 1024),
            (8192, 8192, 1024),
            (16384, 8192, 1024),
            # tune
            (1552, 8192, 8192),
            (1664, 8192, 8192),
            (1792, 8192, 8192),
            (1920, 8192, 8192),
            (3072, 8192, 8192),
            (1552, 10240, 8192),
            (1664, 10240, 8192),
            (1792, 10240, 8192),
            (1920, 10240, 8192),
            (3072, 10240, 8192),
            (1552, 57344, 8192),
            (1664, 57344, 8192),
            (1792, 57344, 8192),
            (1920, 57344, 8192),
            (3072, 57344, 8192),
            (1552, 8192, 28672),
            (1664, 8192, 28672),
            (1792, 8192, 28672),
            (1920, 8192, 28672),
            (3072, 8192, 28672),
            # more shapes
            (16384, 8192, 1280),
            (16384, 1024, 8192),
            (16384, 8192, 7168),
            (16384, 3584, 8192),
            (8192, 8192, 8192)
        ]
        
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="config input of test",
        )
        parser.add_argument(
            "-d",
            "--dtype",
            type=str,
            choices=l_dtype,
            nargs="?",
            const=None,
            default=None,
            help="""Data type.
            e.g.: -d bf16""",
        )
        parser.add_argument(
            "-mnk",
            "--shape",
            type=dtypes.str2tuple,
            nargs="?",
            const=None,
            default=None,
            help="""Shape of mnk.
            e.g. -mnk 1280,8192,1024""",
        )
        
        args = parser.parse_args()
        if args.dtype is None:
            l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
        else:
            l_dtype = [dtypes.d_dtypes[args.dtype]]
        if args.shape is not None:
            l_mnk = [args.shape]
        
        df = []
        for dtype in l_dtype:
            for m, n, k in l_mnk:
                ret = test_gemm(dtype, m, n, k)
                df.append(ret)
        df = pd.DataFrame(df)
        aiter.logger.info(f"summary:\n{df}")
        
        device_name = torch.cuda.get_device_name(0).replace(' ', '_')
        save_file = f"aiter_a4w4_{device_name}_results.csv"
        df.to_csv(save_file)
        aiter.logger.info(f"Saved results to {save_file}")
        
        
        torch.set_default_device("cuda")
        torch.set_printoptions(sci_mode=False)
        SCALE_GROUP_SIZE = 32
        pd.set_option("display.max_columns", 30)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", 30)


@perftest(num_iters=10)
def run_torch(x, w, x_scales, w_scales, dtype):
    m, k = x.shape
    n, k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]

@benchmark()
def test_gemm(dtype, M, N, K):
    ret = {}
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))
    out1 = torch.empty(M, N, dtype=dtype)
    out2 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    out3 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    bias_f32 = None
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)

    a, avg_a = run_torch(x, w, x_scales, w_scales, dtype)

    c, us = run_perftest(
        aiter.gemm_a4w4,
        x,
        wshuffle,
        x_scales_shuffle,
        w_scales_shuffle,
        out2,
        bpreshuffle=True,
    )
    err = checkAllclose(a, c[:M], msg="unified api")
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = err

    return ret

