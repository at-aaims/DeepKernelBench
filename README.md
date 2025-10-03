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

## Run benchmarks
README files in attention, geometrics, matrix_multiply, neural_operator provide the commands to run the benchmarks.

## Support matrix
| Benchmark name | Intel Max1100 | AMD MI250X | AMD MI300A | NVIDIA H100 |
|---|---|---|---|---|
| att  | ❌ | ✅ |✅ | ✅ |
| att2 | ❌| ✅ |✅ | ✅ |
| sdpa | limited| ✅ |✅ | ✅ |
| flex | ❌ | ✅ |✅ | ✅ |
| ao_float8 | ❌ |❌ |✅ | ✅ |
| fp8_gemm | ❌ |❌ |✅ | ✅ |
| semianalysiswork | ❌ |❌ | ✅ |✅|
| bgemm | ✅ |✅ | ✅ |✅ |
| gemm | ✅ |✅ | ✅ |✅ |
| int8_gemm | ✅ |✅ | ✅ |✅ |
| aiter | ❌ |✅ | ✅ |❌|
| tritonBLAS | ❌ |✅ | ✅ |❌|
| vllm | ❌ |❌ | ✅ |✅|
| geometrics kernel | ✅ |✅ | ✅ |✅ |
| neural_operators | ❌ | ✅ |✅ | ✅ |

## Reference
```
https://rocm.blogs.amd.com/artificial-intelligence/flash-attention/README.html
https://github.com/pytorch/ao
https://github.com/ROCm/tritonBLAS
https://github.com/ROCm/aiter
https://github.com/vllm-project/vllm
https://github.com/Dao-AILab/flash-attention
https://github.com/pyg-team/pytorch_geometric
https://github.com/NVIDIA/physicsnemo
https://github.com/neuraloperator/neuraloperator
https://github.com/ORNL/HydraGNN
```
