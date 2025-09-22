## Attention operators
### Benchmark attention operations
```
python driver.py --bench-script attn.benchmark_attn --csv-file attn.shapes.csv
```

### Benchmark Flash attention2 operations
```
cd attn2
python benchmark_attn.py
```

### Benchmark Torch scaled dot product attention operations
```
python driver.py --bench-script sdpa.benchmark_attn --csv-file sdpa.shapes.csv
```

