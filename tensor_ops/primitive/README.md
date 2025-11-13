## Common Torch operators

### Benchmark all operations
```
bash run.sh
```

### To benchmark an operator:
```
python benchmark.py --op <op_name>
```

The Python script also takes optional arguments:
```
--dtype [=fp32 | fp16 | bf16]
--device [=cuda | cpu]
--input-dim dims separated by '-', default "64-1024-1024"
--op-type [=None | binary(for a binary op)]
