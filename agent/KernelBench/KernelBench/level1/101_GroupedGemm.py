import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, X: torch.Tensor, W: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        out = torch._grouped_mm(X, W, offs=offsets)
        return out

B = 4
M = 4096
N = 4096
K = 7168

def gen_grouped_gemm_group_lens(b, m, balance: bool = True):
    """Generate group lengths for grouped GEMM."""
    if balance:
        #Creates a tensor of size (b,) filled with fill_value (m). The tensor’s dtype is inferred from fill_value.
        return torch.full((b,), m, dtype=torch.int64)
    else:
        dist = 0.2 + 0.8 * torch.rand(b)
        dist /= dist.sum()
        group_lens = (dist * b * m).to(torch.int64)
        error = b * m - group_lens.sum()
        group_lens[-1] += error
        return group_lens

def get_inputs():
    X = torch.rand((B*M, K), dtype=torch.bfloat16)
    W = torch.rand((B,K,N),  dtype=torch.bfloat16)
    group_lens = gen_grouped_gemm_group_lens(B, M, balance=True)
    offsets = torch.cumsum(group_lens, dim=0, dtype=torch.int32)
    return [X, W, offsets]

def get_init_inputs():
    return []
