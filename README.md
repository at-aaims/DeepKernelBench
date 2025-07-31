# Quick Start

## Get Help
`python driver.py --help`

## Benchmark attention operations
`python driver.py --bench-script attn.benchmark_attn --csv-file attn/shapes.csv`

## Benchmark GEMM operations
`python driver.py --bench-script gemm.benchmark_gemm --csv-file gemm/shapes.csv`  
`python driver.py --bench-script gemm.benchmark_gemm --csv-file gemm/llama2-70b-shapes.csv`  
`python driver.py --bench-script fp8_gemm.benchmark_gemm --csv-file fp8_gemm/shapes.csv`  
`python driver.py --bench-script fp8_gemm.benchmark_gemm --csv-file fp8_gemm/llama2-70b-shapes.csv`  

## Benchmark batched GEMM operations
`python driver.py --bench-script bgemm.benchmark_bgemm --csv-file bgemm/shapes.csv`

## Integer GEMM operations
`cd int8_gemm`
`python intmm.py --file_path intmm_shapes.csv`

## Reproduce results published by https://semianalysis.com/
`cd semianalysiswork`
`python matmul.py`

