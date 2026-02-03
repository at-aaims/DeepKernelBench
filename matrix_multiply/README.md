## General matrix multiply operations
### Benchmark general matrix multiply operations
```
python driver.py --bench-script gemm.benchmark_gemm --csv-file gemm/forgeL-shapes.csv
python driver.py --bench-script gemm.benchmark_gemm --csv-file gemm/hydragnn-multibranch-shapes.csv
python driver.py --bench-script gemm.benchmark_gemm --csv-file gemm/llama2-70b-shapes.csv
python driver.py --bench-script gemm.benchmark_gemm --csv-file gemm/afnonet-shapes.csv
python driver.py --bench-script fp8_gemm.benchmark_gemm --csv-file fp8_gemm/forgeL-shapes.csv
python driver.py --bench-script fp8_gemm.benchmark_gemm --csv-file fp8_gemm/llama2-70b-shapes.csv
```

### Benchmark batched GEMM operations
```
python driver.py --bench-script bgemm.benchmark_bgemm --csv-file bgemm/shapes.csv
```

### Benchmark integer GEMM operations
```
cd int8_gemm
python intmm.py --file_path intmm_shapes.csv
```

### Reproduce results published by https://semianalysis.com/
```
cd semianalysiswork
python matmul.py
```

### Benchmark forward/backward of `Linear` and `Float8Linear` on LLaMa 2 70B shapes
```
cd ao_float8
python bench_linear_float8.py -o linear_float8_llama70.txt --shape_gen_name llama
```

### Benchmark forward/backward of `Linear` and `Float8Linear` on ForgeL shapes
```
cd ao_float8
python bench_linear_float8.py -o linear_float8_forgeL.txt --shape_gen_name forgeL
```

### Benchmark grouped GEMM operations in BF16 on various MoE models
```
cd grouped_gemm
python bench_grouped_gemm_torch.py
```

### Benchmark Primus Turbo grouped GEMM operations in BF16 and FP8 on various MoE models
```
cd grouped_gemm
python bench_grouped_gemm_turbo.py --dtype fp8 --granularity tensorwise
python bench_grouped_gemm_turbo.py --dtype fp8 --granularity rowwise
python bench_grouped_gemm_turbo.py --dtype fp8 --granularity blockwise
python bench_grouped_gemm_turbo.py --dtype bf16 --backend CK
python bench_grouped_gemm_turbo.py --dtype bf16 --backend HIPBLASLT
```

### Benchmark the GEMM operations in TritonBLAS
```
cd triton_gemm/tritonBLAS
python matmul.py
```

### Benchmark the GEMM operations in AITER
```
cd triton_gemm/aiter
python matmul_fp4.py
python matmul_fp8.py
python matmul.py
```

### Benchmark the FP8-GEMM operations in VLLM
```
cd triton_gemm/vllm
python matmul.py
```

