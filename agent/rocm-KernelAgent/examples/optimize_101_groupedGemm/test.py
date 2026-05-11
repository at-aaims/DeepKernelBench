import torch
import torch.nn as nn

# Summary:
# Reference problem computes grouped GEMM: out = torch._grouped_mm(X, W, offs=offsets)
# with shapes B=4, M=4096, N=4096, K=7168, dtype=bfloat16.
# X: (B*M, K), W: (B, K, N), offsets: cumsum of group lengths (int32).
# The test instantiates a reference Model using the same semantics and compares the kernel_function output.

class ReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, W: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # Use torch._grouped_mm if available; otherwise, fall back to a manual grouped matmul
        if hasattr(torch, "_grouped_mm"):
            return torch._grouped_mm(X, W, offs=offsets)
        else:
            # Manual grouped MM using offsets to split X and select corresponding W[b]
            outs = []
            prev = 0
            # Compute group lengths from offsets
            lens = offsets.to(torch.long).cpu()
            for b in range(int(lens.numel())):
                end = int(lens[b].item())
                X_b = X[prev:end, :]
                W_b = W[b, :, :]
                outs.append(X_b @ W_b)
                prev = end
            return torch.cat(outs, dim=0)


def gen_grouped_gemm_group_lens(b, m, balance: bool = True):
    if balance:
        return torch.full((b,), m, dtype=torch.int64)
    else:
        dist = 0.2 + 0.8 * torch.rand(b)
        dist /= dist.sum()
        group_lens = (dist * b * m).to(torch.int64)
        error = b * m - group_lens.sum()
        group_lens[-1] += error
        return group_lens


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

        # Exact problem specifications
        B = 4
        M = 4096
        N = 4096
        K = 7168
        dtype = torch.bfloat16  # BF16 as specified

        # Create test data using EXACT specifications from problem description
        torch.manual_seed(0)
        X = torch.rand((B * M, K), dtype=dtype, device=device)
        W = torch.rand((B, K, N), dtype=dtype, device=device)
        group_lens = gen_grouped_gemm_group_lens(B, M, balance=True)  # on CPU as in the description
        # Offsets per description: int32 cumsum on CPU
        offsets = torch.cumsum(group_lens, dim=0, dtype=torch.int32)

        print("Input summary:")
        print(f"- X shape: {tuple(X.shape)}, dtype: {X.dtype}, device: {X.device}")
        print(f"- W shape: {tuple(W.shape)}, dtype: {W.dtype}, device: {W.device}")
        print(f"- offsets shape: {tuple(offsets.shape)}, dtype: {offsets.dtype}, device: {offsets.device}")
        print(f"- offsets (first few): {offsets[:min(10, offsets.numel())].tolist()}")

        # Reference computation
        ref_model = ReferenceModel().to(device)
        with torch.no_grad():
            y_ref = ref_model(X, W, offsets)
        if not isinstance(y_ref, torch.Tensor):
            print("Reference model did not return a tensor.")
            return False

        print(f"Reference output shape: {tuple(y_ref.shape)}, dtype: {y_ref.dtype}, device: {y_ref.device}")

        # Call kernel_function as a normal Python function
        with torch.no_grad():
            y = kernel_function(X, W, offsets)

        # Basic checks
        if not isinstance(y, torch.Tensor):
            print("kernel_function did not return a torch.Tensor")
            return False

        # Device check per requirements
        if y.device != X.device:
            print(f"Device mismatch: result device {y.device} vs input device {X.device}")
            return False

        # Shape check
        if y.shape != y_ref.shape:
            print(f"Shape mismatch: got {tuple(y.shape)} vs expected {tuple(y_ref.shape)}")
            return False

        # Numerical verification
        # Convert to float32 for difference computation to avoid BF16 rounding artifacts
        y_f32 = y.to(torch.float32)
        y_ref_f32 = y_ref.to(torch.float32)

        # Adjust tolerances for BF16 and large accumulation dimension (K=7168)
        # BF16 has lower precision; with large K, accumulation error can be higher.
        rtol = 2e-2
        atol = 5e-2

        try:
            allclose = torch.allclose(y_f32, y_ref_f32, rtol=rtol, atol=atol)
        except Exception as e:
            print(f"allclose comparison raised an exception: {e}")
            # Proceed to print debug info and fail
            allclose = False

        if not allclose:
            diff = y_f32 - y_ref_f32
            max_abs = torch.max(torch.abs(diff)).item()
            denom = torch.clamp(torch.abs(y_ref_f32), min=1e-8)
            rel_err = torch.max(torch.abs(diff) / denom).item()
            print("NUMERICAL MISMATCH:")
            print(f"- X shape: {tuple(X.shape)}, dtype: {X.dtype}, device: {X.device}")
            print(f"- W shape: {tuple(W.shape)}, dtype: {W.dtype}, device: {W.device}")
            print(f"- offsets: {offsets.tolist()}")
            print(f"- Expected shape: {tuple(y_ref.shape)}, dtype: {y_ref.dtype}, device: {y_ref.device}")
            print(f"- Result shape: {tuple(y.shape)}, dtype: {y.dtype}, device: {y.device}")
            print(f"- Expected (first 10): {y_ref_f32.flatten()[:10].tolist()}")
            print(f"- Got (first 10): {y_f32.flatten()[:10].tolist()}")
            print(f"- Max absolute difference: {max_abs}")
            print(f"- Max relative error: {rel_err}")
            print(f"- Used tolerances: rtol={rtol}, atol={atol}")
            return False
        else:
            print("Numerical check passed within tolerances.")

        # Additional sanity checks
        if torch.isnan(y).any():
            print("Result contains NaNs.")
            return False
        if torch.isinf(y).any():
            print("Result contains Infs.")
            return False

        return True  # if successful
    except NameError as e:
        print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)