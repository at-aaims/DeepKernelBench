import torch

# Summary:
# Reference model performs scaled dot-product attention:
# out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
# Shapes:
#   batch_size = 32
#   num_heads = 32
#   sequence_length = 512
#   embedding_dimension = 1024
# Inputs must be CUDA tensors. Compare kernel_function(Q, K, V) to reference.

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)


def _select_dtype():
    # Prefer bfloat16 to reduce memory footprint as per requirements; fallback to float16 if bf16 unsupported.
    try:
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _print_tensor_summary(name, t):
    try:
        print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
              f"min={float(t.min()) if t.numel()>0 else 'n/a'}, max={float(t.max()) if t.numel()>0 else 'n/a'}")
        flat = t.flatten()
        n = min(10, flat.numel())
        if n > 0:
            print(f"{name} sample first {n} values: {flat[:n].tolist()}")
    except Exception as e:
        print(f"Could not summarize tensor {name}: {e}")


def test_kernel():
    """Test the kernel implementation."""
    try:
        from kernel import kernel_function
        # Sanity check: kernel should be callable and self-contained
        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        # Device setup
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        device = torch.device("cuda")

        # Exact specifications from problem description
        batch_size = 32
        num_heads = 32
        sequence_length = 512
        embedding_dimension = 1024

        # Choose dtype (prefer bf16 to reduce memory), set tolerances accordingly
        dtype = _select_dtype()
        if dtype == torch.bfloat16:
            rtol, atol = 1e-2, 2e-2  # bfloat16 typically needs looser tolerances
            print("Using bfloat16 for inputs/outputs (preferred). Tolerances set to rtol=1e-2, atol=2e-2.")
        else:
            rtol, atol = 1e-2, 1e-2  # float16 tolerances
            print("bfloat16 not supported; using float16. Tolerances set to rtol=1e-2, atol=1e-2.")

        # Deterministic random inputs (non-zero)
        torch.manual_seed(0)
        try:
            Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device=device, dtype=dtype)
            K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device=device, dtype=dtype)
            V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device=device, dtype=dtype)
        except RuntimeError as oom:
            print(f"Failed to allocate input tensors on device {device}: {oom}")
            return False

        # Print input summaries for debugging
        _print_tensor_summary("Q", Q)
        _print_tensor_summary("K", K)
        _print_tensor_summary("V", V)

        # Instantiate reference model and compute reference output on CUDA
        model = Model().eval().to(device=device)
        with torch.no_grad():
            # Prefer memory-efficient SDP kernels to reduce peak memory
            try:
                cm = torch.backends.cuda.sdp_kernel
                # Enable mem-efficient path; flash may not support head_dim=1024, so keep it optional
                with cm(enable_flash=False, enable_mem_efficient=True, enable_math=False):
                    y_ref = model(Q, K, V)
            except Exception:
                # Fallback: let PyTorch choose automatically
                y_ref = model(Q, K, V)

        # Basic reference checks
        if not isinstance(y_ref, torch.Tensor):
            print("Reference model did not return a torch.Tensor")
            return False
        if y_ref.device != Q.device:
            print(f"Reference output device mismatch: expected {Q.device}, got {y_ref.device}")
            return False

        # Call kernel_function as a normal Python function
        print("Calling kernel_function(Q, K, V)...")
        with torch.no_grad():
            y_kernel = kernel_function(Q, K, V)

        # Validate types and devices
        if not isinstance(y_kernel, torch.Tensor):
            print(f"kernel_function returned non-tensor type: {type(y_kernel)}")
            return False

        if y_kernel.device != Q.device:
            print(f"kernel_function output device mismatch: expected {Q.device}, got {y_kernel.device}")
            return False

        # Validate shapes
        if y_kernel.shape != y_ref.shape:
            print(f"Shape mismatch: expected {tuple(y_ref.shape)}, got {tuple(y_kernel.shape)}")
            return False

        # Check for NaNs/Infs
        if not torch.isfinite(y_kernel).all():
            n_nan = torch.isnan(y_kernel).sum().item()
            n_inf = torch.isinf(y_kernel).sum().item()
            print(f"kernel_function output contains non-finite values: NaNs={n_nan}, Infs={n_inf}")
            return False

        # Numerical comparison: cast to float32 for a stable comparison metric
        y_ref_32 = y_ref.to(torch.float32)
        y_kernel_32 = y_kernel.to(torch.float32)

        # Verify results with detailed debugging on failure
        try:
            close = torch.allclose(y_kernel_32, y_ref_32, rtol=rtol, atol=atol)
            if not close:
                diff = (y_kernel_32 - y_ref_32)
                max_abs = torch.max(torch.abs(diff)).item()
                mean_abs = torch.mean(torch.abs(diff)).item()
                # To avoid division by near zero, add small epsilon
                rel = torch.max(torch.abs(diff) / (torch.abs(y_ref_32) + 1e-8)).item()
                print("NUMERICAL MISMATCH:")
                print(f"Input shapes: Q {tuple(Q.shape)}, K {tuple(K.shape)}, V {tuple(V.shape)}")
                print(f"Input dtypes: Q {Q.dtype}, K {K.dtype}, V {V.dtype}")
                print(f"Output shape: {tuple(y_kernel.shape)}, dtype: {y_kernel.dtype}")
                print(f"Expected shape: {tuple(y_ref.shape)}, dtype: {y_ref.dtype}")
                print(f"Expected (first few): {y_ref_32.flatten()[:10].tolist()}")
                print(f"Got (first few): {y_kernel_32.flatten()[:10].tolist()}")
                print(f"Max absolute difference: {max_abs}")
                print(f"Mean absolute difference: {mean_abs}")
                print(f"Max relative error: {rel}")
                # Extra diagnostics: norms
                print(f"||expected||_inf: {float(y_ref_32.abs().max())}, ||got||_inf: {float(y_kernel_32.abs().max())}")
                return False
        except Exception as e:
            print(f"Error during numerical comparison: {e}")
            # Print a few more details for debugging
            print(f"y_ref summary: shape={tuple(y_ref.shape)}, dtype={y_ref.dtype}, device={y_ref.device}")
            print(f"y_kernel summary: shape={tuple(y_kernel.shape)}, dtype={y_kernel.dtype}, device={y_kernel.device}")
            return False

        print("Test passed: kernel_function output matches reference within tolerance.")
        return True  # if successful

    except NameError as e:
        # Surface undefined helper issues from kernel.py clearly
        print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        return False
    except RuntimeError as e:
        # Commonly OOM or CUDA errors
        print(f"Test failed: RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)