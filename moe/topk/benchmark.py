import torch

from topk import topk, topk_torch

def run_benchmark(num_rows, num_cols, k, dtype,
                  apply_softmax=True,
                  warmup_iter=30, num_iter=1000,
                  log=True, profile=False):

    torch.manual_seed(0)
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dtype = getattr(torch, dtype)
    x = torch.randn((num_rows, num_cols), dtype=dtype, device=device)

    for _ in range(warmup_iter):
        sparse_x_tri = topk(x, k, apply_softmax=apply_softmax)

    torch.cuda.synchronize(device=device)

    sparse_x_ref = topk_torch(x, k, apply_softmax=apply_softmax)
    torch.allclose(sparse_x_tri.vals, sparse_x_ref.vals)
    torch.equal(sparse_x_tri.indx, sparse_x_ref.indx)
    torch.equal(sparse_x_tri.mask.storage.data, sparse_x_ref.mask.storage.data)
    assert sparse_x_tri.mask.storage.data.stride() == sparse_x_ref.mask.storage.data.stride()
    assert sparse_x_tri.mask.storage.data.shape == sparse_x_ref.mask.storage.data.shape

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    begin.record()

    for _ in range(num_iter):
        sparse_x_tri = topk(x, k, apply_softmax=apply_softmax)

    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0  # total iteration time

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_gb = peak_bytes / (1024**3)

    print(f"{num_iter / time:.6f} iter/s, {time:.3f} sec, {peak_mem_gb:.3f} GiB")
    return time

# num_cols < 32768
if __name__ == "__main__":
    for b in [4, 16, 64]:
        for k in [8, 16, 32]:
            for c in [2**10, 2**12, 2**14]:
                print(f"num_rows={b}, num_cols={c}, k={k}")
                run_benchmark(b, c, k, 'bfloat16')
