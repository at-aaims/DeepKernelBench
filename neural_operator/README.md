## Neural operators
### Benchmark Fourier neural operations in neural_operator
```
python driver.py --bench-script neural_operator.benchmark_fno --csv-file neural_operator/fno_shapes.csv
python driver.py --bench-script neural_operator.benchmark_sfno --csv-file neural_operator/fno_shapes.csv
```

### Benchmark Fourier neural operations in PhysicsNeMo
```
python driver.py --bench-script neural_operator.benchmark_physicsnemo_fno --csv-file neural_operator/fno_shapes.csv
```
