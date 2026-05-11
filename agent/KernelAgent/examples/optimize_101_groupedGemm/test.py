import sys
import traceback
import torch
import torch.nn as nn

# Summary:
# This test validates a Triton kernel that should compute a grouped GEMM equivalent to:
# out = torch._grouped_mm(X, W, offs=offsets)
# with shapes/dtypes:
# - B=4, M=4096, N=4096, K=7168
# - X: (B*M, K), bfloat16
# - W: (B, K, N), bfloat16
# - offsets: cumsum of group lengths (int32), shape (B,)
#
# The test:
# - Creates inputs exactly as specified on CUDA
# - Computes a PyTorch reference using Model(X, W, offsets)
# - Calls kernel_function(X, W, offsets) as a normal Python function
# - Compares outputs with tolerances appropriate for bf16 and large accumulation
# - Provides detailed debugging output on failure
# - Returns True on success, False otherwise, and exits with code 0/1


def test_kernel():
    """Test the kernel implementation for grouped GEMM using torch._grouped_mm as reference."""
    try:
        try:
            from kernel import kernel_function
        except Exception as e:
            print("Failed to import kernel_function from kernel.py")
            traceback.print_exc()
            return False

        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        if not torch.cuda.is_available():
            print("CUDA not available on this system; this test requires a CUDA device.")
            return False

        device = torch.device("cuda")
        print(f"Using device: {device}, CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")

        # Exact problem specs
        B = 4
        M = 4096
        N = 4096
        K = 7168

        # Define reference model using torch._grouped_mm per the problem description
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, X: torch.Tensor, W: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
                out = torch._grouped_mm(X, W, offs=offsets)
                return out

        # Helper to generate group lengths exactly as given
        def gen_grouped_gemm_group_lens(b, m, balance: bool = True):
            if balance:
                # Creates a tensor of size (b,) filled with m
                return torch.full((b,), m, dtype=torch.int64)
            else:
                dist = 0.2 + 0.8 * torch.rand(b)
                dist /= dist.sum()
                group_lens = (dist * b * m).to(torch.int64)
                error = b * m - group_lens.sum()
                group_lens[-1] += error
                return group_lens

        # Construct inputs exactly as specified, on CUDA device with bfloat16 dtype
        # Note: These tensors are large; ensure the GPU has sufficient memory.
        try:
            # X: (B*M, K) bf16
            X = torch.rand((B * M, K), dtype=torch.bfloat16, device=device)
            # W: (B, K, N) bf16
            W = torch.rand((B, K, N), dtype=torch.bfloat16, device=device)
        except RuntimeError as oom:
            print("Out of memory when allocating X/W. Ensure sufficient GPU memory.")
            traceback.print_exc()
            return False

        # Offsets: cumsum of group lengths (int32), shape (B,)
        # Create group lengths on CPU (small), then prepare offsets
        group_lens_cpu = gen_grouped_gemm_group_lens(B, M, balance=True)  # int64 on CPU
        offsets_cpu = torch.cumsum(group_lens_cpu, dim=0, dtype=torch.int32)  # int32 on CPU
        # For kernel, we will use offsets on device to match X/W; for reference, we will try GPU first, then CPU if needed
        offsets = offsets_cpu.to(device=device, dtype=torch.int32)

        # Initialize reference model
        model = Model().to(device=device)

        # Compute reference output
        # Some PyTorch builds may require offs on CPU; try GPU first, then fallback
        try:
            y_ref = model(X, W, offsets)
        except RuntimeError as e_gpu_offs:
            print(f"Encountered RuntimeError using GPU offsets for reference: {e_gpu_offs}")
            print("Retrying reference computation with CPU offsets for offs argument...")
            try:
                y_ref = model(X, W, offsets_cpu)
            except Exception as e_cpu_offs:
                print("Failed to compute reference output even with CPU offsets.")
                traceback.print_exc()
                return False
        except Exception as e:
            print("Unexpected error during reference computation:")
            traceback.print_exc()
            return False

        if not isinstance(y_ref, torch.Tensor):
            print("Reference output is not a torch.Tensor")
            return False

        if y_ref.device.type != 'cuda':
            print(f"Reference output not on CUDA device. Got: {y_ref.device}")
            return False

        expected_shape = (B * M, N)
        if y_ref.shape != expected_shape:
            print(f"Reference output has incorrect shape. Expected {expected_shape}, got {y_ref.shape}")
            return False

        if y_ref.dtype != torch.bfloat16:
            print(f"Reference output has incorrect dtype. Expected bfloat16, got {y_ref.dtype}")
            return False

        torch.cuda.synchronize()

        # Call the Triton-backed kernel_function as a normal Python function
        try:
            y_kernel = kernel_function(X, W, offsets)
        except NameError as ne:
            print("Test failed: NameError (likely undefined helper in kernel.py):", ne)
            traceback.print_exc()
            return False
        except Exception as e:
            print("kernel_function raised an exception during execution:")
            traceback.print_exc()
            return False

        if not isinstance(y_kernel, torch.Tensor):
            print(f"kernel_function returned a non-tensor object of type: {type(y_kernel)}")
            return False

        # Device check: result should be on the same device as inputs
        if y_kernel.device != X.device:
            print(f"Device mismatch: kernel output device {y_kernel.device}, input device {X.device}")
            return False

        # Basic shape/dtype checks
        if y_kernel.shape != expected_shape:
            print(f"Output shape mismatch. Expected {expected_shape}, got {y_kernel.shape}")
            return False

        if y_kernel.dtype != torch.bfloat16:
            print(f"Output dtype mismatch. Expected bfloat16, got {y_kernel.dtype}")
            return False

        torch.cuda.synchronize()

        # Numerical comparison
        # For bf16 and large accumulation (K=7168), allow slightly looser tolerances
        # Justification: bf16 has limited mantissa and matmuls over large K accumulate rounding errors.
        rtol = 1e-2
        atol = 2e-2

        try:
            close = torch.allclose(y_kernel, y_ref, rtol=rtol, atol=atol)
        except Exception as e:
            print("Error during numerical comparison with torch.allclose:")
            traceback.print_exc()
            return False

        if not close:
            # Compute detailed stats for debugging
            diff = (y_kernel - y_ref).to(torch.float32)
            abs_diff = torch.abs(diff)
            max_abs_diff = abs_diff.max().item()
            # Avoid division by zero by adding small epsilon
            denom = torch.abs(y_ref.to(torch.float32)) + 1e-8
            rel_err = (abs_diff / denom).max().item()

            print("NUMERICAL MISMATCH between kernel output and reference.")
            print(f"Allowed rtol={rtol}, atol={atol}")
            print(f"Max absolute difference: {max_abs_diff}")
            print(f"Max relative error: {rel_err}")
            print(f"Input X shape: {X.shape}, dtype: {X.dtype}, device: {X.device}")
            print(f"Input W shape: {W.shape}, dtype: {W.dtype}, device: {W.device}")
            print(f"Offsets shape: {offsets.shape}, dtype: {offsets.dtype}, device: {offsets.device}")
            print(f"Reference output shape: {y_ref.shape}, dtype: {y_ref.dtype}, device: {y_ref.device}")
            print(f"Kernel output shape: {y_kernel.shape}, dtype: {y_kernel.dtype}, device: {y_kernel.device}")

            # Show sample values for first few elements (flattened)
            num_print = 10
            y_ref_fp32 = y_ref.to(torch.float32).flatten()
            y_kernel_fp32 = y_kernel.to(torch.float32).flatten()
            print(f"Reference (first {num_print}): {y_ref_fp32[:num_print]}")
            print(f"Kernel    (first {num_print}): {y_kernel_fp32[:num_print]}")
            # Indices of largest absolute differences
            topk = min(5, abs_diff.numel())
            topk_vals, topk_idx = torch.topk(abs_diff.flatten(), k=topk)
            print("Top-5 absolute diffs (val, idx):")
            for i in range(topk):
                idx = topk_idx[i].item()
                print(f"  idx {idx}: ref={y_ref_fp32[idx].item()}, got={y_kernel_fp32[idx].item()}, abs_diff={topk_vals[i].item()}")
            return False

        print("Test passed: kernel output matches reference within tolerances.")
        return True

    except NameError as ne:
        # Surface undefined helper issues from kernel.py clearly
        print(f"Test failed: NameError (likely undefined helper in kernel.py): {ne}")
        traceback.print_exc()
        return False
    except Exception as e:
        print("Test failed due to unexpected exception:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)