"""
Fused Softmax
"""

from typing import Optional

import torch
import triton
from triton.runtime import driver
from triton_softmax import softmax
from triton_softmax2 import softmax2

DEVICE = driver.active.get_active_torch_device()

def get_benchmark(providers_filter: Optional[list[str]] = None):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["N"],  # argument names to use as an x-axis for the plot
            x_vals=[256, 1024, 2048, 4096, 1024 * 8, 1024 * 16, 1024 * 32],  # different possible values for `x_name`
            line_arg="provider",  # argument name whose value corresponds to a different line in the plot
            line_vals=['triton_softmax1', 'triton_softmax2', 'torch'],  # possible values for `line_arg``
            line_names=['Triton_softmax1', 'Triton_softmax2', 'Torch'],  # label name for the lines
            styles=[("blue", "-"), ("red", "-"), ("green", "-")],  # line styles
            ylabel=["GB/s"],  # label name for the y-axis
            plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
            args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
        ))

    def benchmark(M, N, provider):
        x = torch.randn(M, N, device=DEVICE, dtype=torch.bfloat16)
        if provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        elif provider == "triton_softmax1":
            triton_fn = lambda: softmax(x)
            ms = triton.testing.do_bench(triton_fn)
        elif provider == "triton_softmax2":
            triton_fn = lambda: softmax2(x)
            ms = triton.testing.do_bench(triton_fn)

        else:
            raise NotImplementedError(f"Unsupported provider {provider}")
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    return benchmark


if __name__ == "__main__":
    # a unit test
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    y_torch = torch.softmax(x, axis=1)

    y_triton = softmax(x)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    y_triton = softmax2(x)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    _benchmark = get_benchmark()
    _benchmark.run(show_plots=False, print_data=True)
