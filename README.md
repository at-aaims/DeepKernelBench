# Quick Start

## NVIDIA GPUs
1. Download and install the CUDA Toolkit for your corresponding platform. For system requirements and installation instructions of the CUDA toolkit, please refer to the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
2. Create a new virtual env and install build dependencies
```
   pip install -r requirements-cuda.txt
```

## AMD GPUs
1. Download and install the ROCm Toolkit for your corresponding platform. For system requirements and installation instructions of ROCm toolkit, please refer to the [Linux Installation Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
2. Create a new virtual env and install build dependencies
```
   pip install -r requirements-rocm.txt
```

## Intel XPU devices
1. Download and install the [Intel oneAPI 2025.3 Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
2. Create a new virtual env and install build dependencies
```
   pip install -r requirements-xpu.txt
```


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
| group-gemm-triton | ✅ |✅ | ✅ |✅ |
| group-gemm-turbo | ❌ | ❌ | ✅ |❌ |
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
| storeKVCache | ✅ |✅ | ✅ |✅ |
| mixtral-moe | ❌ | ✅ | ✅ |✅ |

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
