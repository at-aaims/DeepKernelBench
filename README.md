## Quick Start

# Get Help
python driver.py --help

# Benchmark attention operations
python driver.py --bench-script attn.benchmark_attn --csv-file attn/shapes.csv

# Benchmark GEMM operations
python driver.py --bench-script gemm.benchmark_gemm --csv-file gemm/shapes.csv

# Benchmark batched GEMM operations
python driver.py --bench-script bgemm.benchmark_bgemm --csv-file bgemm/shapes.csv

