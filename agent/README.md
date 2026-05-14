`KernelAgent` turns PyTorch programs into verified Triton kernels and optimize its performance on NVIDIA GPUs. It was designed around KernelBench workloads. Please see README.md for more details.

`rocm-KernelAgent` turns PyTorch programs into verified Triton kernels and optimize its performance on AMD GPUs. It was designed around KernelBench workloads. Please see README.md for more details.

`KernelBench` contains many workloads. Please see README.md for more details.

`optimized-kernels` contains the scripts to run the Triton kernels (SDPA and grouped GEMM) generated from KernelAgent and rocm-KernelAgent. Users need to manually select the CUDA or ROCm kernels in the following scripts for performance evaluation.

To evaluate the performance of the SDPA Triton kernels
```
python benchmark_attn_kernel_agent.py
```

To evaluate the performance of the Grouped GEMM Triton kernels
```
python bench_grouped_gemm_kernel_agent.py
```
