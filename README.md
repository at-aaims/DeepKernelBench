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
The README files in sub-directories provide the commands to run the benchmarks.

## Support matrix
| Benchmark name | Intel B580 (16GB) | AMD MI250X (64GB) | AMD MI300A (128GB) | NVIDIA H100 (80GB) |
|---|---|---|---|---|
| attn  | ❌ | ✅ |✅ | ✅ |
| attn_triton | ✅ | ✅ |✅ | ✅ |
| sdpa | limited| ✅ |✅ | ✅ |
| flex | ✅ | ✅ |✅ | ✅ |
| fp8_gemm | ✅ |❌ |✅ | ✅ |
| semianalysiswork | ✅ |❌ | ✅ |✅|
| bgemm | ✅ |✅ | ✅ |✅ |
| group-gemm | ❌ |✅ | ✅ |✅ |
| gemm | ✅ |✅ | ✅ |✅ |
| int8_gemm | ✅ |✅ | ✅ |✅ |
| aiter | ❌ |✅ | ✅ |❌|
| tritonBLAS | ❌ |✅ | ✅ |❌|
| vllm | ❌ |❌ | ✅ |✅|
| geometrics kernel | ✅ |✅ | ✅ |✅ |
| neural_operators | ❌ | ✅ |✅ | ✅ |
| tensor_ops | ✅ |✅ | ✅ |✅ |
| communication | ✅ |✅ | ✅ |✅ |
| moe | ✅ |✅ | ✅ |✅ |

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
