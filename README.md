# Quick Start

## Install Flash Attention with ROCm support 
```
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention
MAX_JOBS=$((`nproc` - 1)) pip install -v .
```

## Get Help
```
python driver.py --help
```

## Benchmark attention operations
```
python driver.py --bench-script attn.benchmark_attn --csv-file attn/shapes.csv
```

## Benchmark Flash attention2 operations
```
cd attn2
python benchmark_attn.py
```

## Benchmark Torch scaled dot product attention operations
```
python driver.py --bench-script sdpa.benchmark_attn --csv-file attn/shapes.csv
```

## Benchmark GEMM operations
```
python driver.py --bench-script gemm.benchmark_gemm --csv-file gemm/shapes.csv
python driver.py --bench-script gemm.benchmark_gemm --csv-file gemm/llama2-70b-shapes.csv
python driver.py --bench-script fp8_gemm.benchmark_gemm --csv-file fp8_gemm/shapes.csv
python driver.py --bench-script fp8_gemm.benchmark_gemm --csv-file fp8_gemm/llama2-70b-shapes.csv
```

## Benchmark batched GEMM operations
```
python driver.py --bench-script bgemm.benchmark_bgemm --csv-file bgemm/shapes.csv
```

## Benchmark integer GEMM operations
```
cd int8_gemm
python intmm.py --file_path intmm_shapes.csv
```

## Reproduce results published by https://semianalysis.com/
```
cd semianalysiswork
python matmul.py
```

## Benchmark forward/backward of `Linear` and `Float8Linear` on LLaMa 2 70B shapes
```
cd ao_float8
python bench_linear_float8.py -o linear_float8_llama70.txt --shape_gen_name llama
```

## Benchmark forward/backward of `Linear` and `Float8Linear` on ForgeL shapes
```
cd ao_float8
python bench_linear_float8.py -o linear_float8_forgeL.txt --shape_gen_name forgeL
```

## Benchmark grouped GEMM operations on LLaMa 4 shapes
```
cd ao_float8
python bench_grouped_mm.py
```

## Benchmark the GEMM operations in TritonBLAS
```
cd triton_gemm/tritonBLAS
python matmul.py
```

## Benchmark the GEMM operations in AITER
```
cd triton_gemm/aiter
python matmul.py
```

## Benchmark the FP8-GEMM operations in VLLM
```
cd triton_gemm/vllm
python matmul.py
```

## Benchmark Torch Geometric operations
```
cd geometric/kernel
python benchmark_kernel.py --layers 4 --hiddens 866 --epochs 100 --batch_size 128 --inference --compile
```

# Reference
```
https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html
https://github.com/pytorch/ao
https://github.com/ROCm/tritonBLAS
https://github.com/ROCm/aiter
https://github.com/vllm-project/vllm
https://github.com/Dao-AILab/flash-attention
```
