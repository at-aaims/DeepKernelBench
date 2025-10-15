## Distributed communication operations

### Gathers tensors from the whole group in a list
```
torchrun --standalone --nnodes 1 --nproc-per-node 2 benchmark_allgather.py
```

### Reduces the tensor data across all machines in a way that all get the final result
```
torchrun --standalone --nnodes 1 --nproc-per-node 2 benchmark_allreduce.py
```

### Reduces, then scatters a list of tensors to all processes in a group
```
torchrun --standalone --nnodes 1 --nproc-per-node 2 benchmark_reduce_scatter.py
```
