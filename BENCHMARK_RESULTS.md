# Benchmark results

## AMD Instinct MI300A

Representative benchmarks were validated on September 28, 2026. These results are
not a full sweep of every optional backend in the support matrix.

### Environment

- 4 x AMD Instinct MI300A (`gfx942`), 101,122,895,872 bytes visible VRAM per device
- `rocm/pytorch:latest` container
- PyTorch 2.10.0 (`rocm7.2.4`), Triton 3.6.0
- Single GPU unless a benchmark says otherwise

### Results

| Benchmark | Configuration | Result |
|---|---|---:|
| BF16 GEMM | 4096 x 4096 x 4096, forward | 188.2 TFLOP/s |
| FP8 GEMM | 4096 x 4096 x 4096, forward | 353.2 TFLOP/s; cosine similarity 0.9961 |
| BF16 batched GEMM | batch 8, 1024 x 1024 x 1024, forward | 0.04 TFLOP/s |
| SDPA math | batch 8, sequence 512, 8 heads x 64, causal forward | 3.1 TFLOP/s |
| SDPA flash | same shape | 37.7 TFLOP/s |
| SDPA memory-efficient | same shape | 34.2 TFLOP/s |
| Triton attention | same shape | 23.4 TFLOP/s; see correctness note |
| Triton softmax | M=4096, N=256 through 32768 | 644.2 to 1412.6 GB/s |
| Triton GELU | M=4096, N=256 through 12672 | up to 1764.8 GB/s |
| Compiled KV-cache store | item size 64 through 1024, batch 1 through 16384 | 1.82 to 231.58 us |
| All-reduce, 2 GPUs | SUM, messages through 10 GB | up to 89.88 GB/s |
| Torch MoE | 32 tokens, tensor parallel size 8 | 0.761 iteration/s; 1.562 GiB peak |

The batched GEMM result is functionally successful but unexpectedly slow for this
shape and software stack. Treat it as a regression signal, not a device ceiling.
The Triton attention correctness check found 38 mismatches among 2,097,152 BF16
outputs; the largest absolute difference was 0.015625 against a 0.01 tolerance.
Its throughput is recorded for diagnosis, but that run did not pass correctness.

### Reproduction note

The pinned wheels in `requirements-rocm.txt` require CPython 3.12. A wheel/runtime
mismatch can import successfully and still fail during GPU execution, so record
the PyTorch ROCm version and host ROCm runtime with published measurements.
