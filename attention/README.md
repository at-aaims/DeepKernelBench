## Attention operators
### Benchmark Flash attention2 operations (flash_attn_func and flash_attn_qkvpacked_func)
```
python driver.py --bench-script attn.benchmark_attn --csv-file attn/shapes.csv
python driver.py --bench-script attn.benchmark_attn2 --csv-file attn/shapes.csv
```

### Benchmark Flash attention3 operations
```
python driver.py --bench-script attn.benchmark_attn3 --csv-file attn/shapes.csv
```

### Benchmark Flash attention4 operations
```
python driver.py --bench-script attn.benchmark_attn4 --csv-file attn/shapes.csv
```

### Benchmark Torch scaled dot product attention operations
```
python driver.py --bench-script sdpa.benchmark_attn --csv-file sdpa/shapes.csv
```

### Benchmark Torch flex attention operations
```
python driver.py --bench-script flex.benchmark_flex --csv-file flex/shapes.csv
```

### Benchmark Triton attention operations
```
python driver.py --bench-script attn_triton.benchmark_attn --csv-file attn/shapes.csv
```

### Benchmark AITER attention operations
```
python driver.py --bench-script attn_aiter.benchmark_attn --csv-file attn/aiter_shapes.csv
```

### Benchmark AITER Flash attention4 (MHA v4)
Dense, head_dim 128, non-causal, forward only. The `recipe` column selects `bf16`, `fp8`, or `mxfp4`. FP8 and MXFP4 time quantization together with the kernel. Kernels ship for gfx942 (MI300) and gfx950 (MI350/MI355).
```
python driver.py --bench-script attn_aiter.benchmark_attn4 --csv-file attn/fa4_aiter_shapes.csv
```

### Benchmark VLLM paged attention v2 operations
```
python driver.py --bench-script attn_paged.benchmark_paged_attention --csv-file attn_paged/shapes.csv
```
