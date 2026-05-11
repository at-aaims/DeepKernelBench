import torch
import torch.nn as nn
import sys
import traceback

# Summary:
# Reference problem computes scaled dot-product attention:
# out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
# Shapes:
#   batch_size = 32
#   num_heads = 32
#   sequence_length = 512
#   embedding_dimension = 1024
# The test creates CUDA tensors (prefer bf16), calls kernel_function(Q, K, V),
# and compares against the PyTorch reference Model output using appropriate tolerances.

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Default: no dropout, not causal; uses PyTorch's optimized kernels when possible
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)


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

        # Set seeds for reproducibility (not strictly required but helps debugging)
        torch.manual_seed(0)

        # Use bf16 when possible (as per requirements), fallback to fp16 if not supported
        preferred_dtype = torch.bfloat16
        alt_dtype = torch.float16
        dtype_used = None

        # Problem-specified exact shapes
        batch_size = 32
        num_heads = 32
        sequence_length = 512
        embedding_dimension = 1024

        # Create test data using EXACT specifications
        # Try bf16 first; if a runtime error occurs due to unsupported dtype, fallback to fp16
        def make_inputs(dtype):
            Q = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device=device, dtype=dtype)
            K = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device=device, dtype=dtype)
            V = torch.rand(batch_size, num_heads, sequence_length, embedding_dimension, device=device, dtype=dtype)
            return Q, K, V

        try:
            Q, K, V = make_inputs(preferred_dtype)
            dtype_used = preferred_dtype
        except RuntimeError as e:
            print(f"bf16 tensor allocation failed with error: {e}")
            print("Falling back to float16 for inputs due to device/dtype limitations.")
            Q, K, V = make_inputs(alt_dtype)
            dtype_used = alt_dtype

        # Instantiate reference model on CUDA
        model = Model().eval().to(device)

        # Compute reference output
        with torch.no_grad():
            y_ref = model(Q, K, V)
        torch.cuda.synchronize(device)

        # Call kernel_function as a normal Python function (no Triton launch syntax here)
        y_kernel = kernel_function(Q, K, V)
        torch.cuda.synchronize(device)

        # If kernel returns a tuple/list, try to use the first tensor element as output
        if isinstance(y_kernel, (tuple, list)):
            if len(y_kernel) == 0:
                print("kernel_function returned an empty tuple/list.")
                return False
            y_kernel = y_kernel[0]

        # Basic checks
        if not isinstance(y_kernel, torch.Tensor):
            print(f"kernel_function returned a non-tensor type: {type(y_kernel)}")
            return False

        # Device check: result should be on same device as input
        if y_kernel.device != Q.device:
            print(f"Device mismatch: result device {y_kernel.device}, input device {Q.device}")
            return False

        # Shape check
        if y_kernel.shape != y_ref.shape:
            print(f"Shape mismatch: expected {y_ref.shape}, got {y_kernel.shape}")
            return False

        # Finite check
        if not torch.isfinite(y_kernel).all():
            print("kernel_function output contains non-finite values (NaN or Inf).")
            # Print a small sample to aid debugging
            flat = y_kernel.flatten()
            print(f"Sample of non-finite check (first 20): {flat[:20].detach().cpu()}")
            return False

        # Numerical comparison
        # Cast to float32 for comparison to reduce dtype mismatch sensitivity
        y_ref_f32 = y_ref.to(torch.float32)
        y_kernel_f32 = y_kernel.to(torch.float32)

        # Choose tolerances based on dtype and problem size (bf16/fp16 need looser tolerances)
        # Default: rtol=1e-3, atol=1e-3
        # For bf16/fp16: use rtol=1e-2, atol=2e-2 (lower precision)
        if dtype_used in (torch.bfloat16, torch.float16):
            rtol = 1e-2
            atol = 2e-2
        else:
            rtol = 1e-3
            atol = 1e-3

        # Perform comparison
        try:
            if not torch.allclose(y_kernel_f32, y_ref_f32, rtol=rtol, atol=atol):
                # Print detailed debugging info
                diff = (y_kernel_f32 - y_ref_f32).abs()
                max_abs = diff.max().item()
                # Relative error per element: abs(err) / (abs(ref)+eps)
                eps = 1e-8
                rel_err = diff / (y_ref_f32.abs() + eps)
                max_rel = rel_err.max().item()

                print("NUMERICAL MISMATCH:")
                print(f"Input shapes: Q={tuple(Q.shape)}, K={tuple(K.shape)}, V={tuple(V.shape)}")
                print(f"Input dtypes: Q={Q.dtype}, K={K.dtype}, V={V.dtype}")
                print(f"Output shape: expected={tuple(y_ref.shape)}, got={tuple(y_kernel.shape)}")
                print(f"Output dtypes: expected={y_ref.dtype}, got={y_kernel.dtype}")
                print(f"Comparison dtype: float32; rtol={rtol}, atol={atol}")
                print(f"Max absolute difference: {max_abs}")
                print(f"Max relative error: {max_rel}")

                # Print small samples from a consistent slice
                b_idx, h_idx, t_idx = 0, 0, 0
                slice_len = min(10, y_ref.shape[-1])
                exp_sample = y_ref_f32[b_idx, h_idx, t_idx, :slice_len].detach().cpu()
                got_sample = y_kernel_f32[b_idx, h_idx, t_idx, :slice_len].detach().cpu()
                print(f"Expected sample [0,0,0,:{slice_len}]: {exp_sample}")
                print(f"Got sample [0,0,0,:{slice_len}]: {got_sample}")

                # Also show a random location sample to diversify debugging
                import random
                rb = random.randint(0, y_ref.shape[0] - 1)
                rh = random.randint(0, y_ref.shape[1] - 1)
                rt = random.randint(0, y_ref.shape[2] - 1)
                exp_sample2 = y_ref_f32[rb, rh, rt, :slice_len].detach().cpu()
                got_sample2 = y_kernel_f32[rb, rh, rt, :slice_len].detach().cpu()
                print(f"Random sample at [{rb},{rh},{rt},: {slice_len}]:")
                print(f"Expected: {exp_sample2}")
                print(f"Got     : {got_sample2}")

                return False
        except Exception as cmp_e:
            print("Exception during numerical comparison:")
            print("".join(traceback.format_exception(type(cmp_e), cmp_e, cmp_e.__traceback__)))
            # Provide some context data for debugging
            print(f"y_ref - dtype: {y_ref.dtype}, device: {y_ref.device}, shape: {y_ref.shape}")
            print(f"y_kernel - dtype: {y_kernel.dtype}, device: {y_kernel.device}, shape: {y_kernel.shape}")
            return False

        # All checks passed
        print("Test passed: kernel_function output matches reference within tolerances.")
        return True

    except NameError as e:
        print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        return False
    except RuntimeError as e:
        # Common failure: OOM or unsupported dtype
        msg = str(e)
        if "out of memory" in msg:
            print("Test failed: CUDA out of memory during test execution.")
        else:
            print(f"Test failed: RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)