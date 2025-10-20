## Attention operators
### Benchmark Flash attention2 operations
```
python driver.py --bench-script attn.benchmark_attn --csv-file attn/shapes.csv
python driver.py --bench-script attn.benchmark_attn2 --csv-file attn/shapes.csv
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
cd triton
python benchmark_attn_triton.py
```
