# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
from dataclasses import dataclass
import torch
import torch.nn.functional as F

def init_compute_data(M, K, N, E, dtype, device):

    randbits = [torch.randperm(E) for _ in range(M)]
    x_list = [
        (-1) ** i
        * ((16384 + ((i * 512) % 4096) + bits).to(torch.int16).view(dtype))
        for i, bits in enumerate(randbits)
    ]
    exp_data = torch.stack(x_list).to(device=device)  # simulating gate_output (M, E)

    # create input tensor
    x = torch.randn((M, K), dtype=dtype, device=device)
    w1 = torch.randn((E, 2 * N, K), dtype=dtype, device=device)
    w1_bias = torch.randn((E, 2 * N), dtype=dtype, device=device)

    w2 = torch.randn((E, K, N), dtype=dtype, device=device)
    w2_bias = torch.randn((E, K), dtype=dtype, device=device)

    return (
        x,
        w1,
        w1_bias,
        w2,
        w2_bias,
        exp_data,
    )


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


def swiglu(x, alpha: float = 1.702, limit: float = 1.0):
    # Note we add an extra bias of 1 to the linear layer
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    if limit is not None:
        x_linear = x_linear.clamp(min=-limit, max=limit)
    return out_glu * (x_linear + 1)


def oai_moe_forward(
    hidden_states: torch.Tensor,  # (M, K)
    w1: torch.Tensor,             # (E, 2N, K)
    w1_bias: torch.Tensor,        # (E, 2N)
    w2: torch.Tensor,             # (E, K, N)
    w2_bias: torch.Tensor,        # (E, K)
    gating_output: torch.Tensor,  # (M, E)
    topk: int,
):
    t = hidden_states
    experts = torch.topk(gating_output, k=topk, dim=-1, sorted=True)
    # softmax over topK elements
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
    expert_indices = experts.indices

    # MLP #1
    mlp1_weight = w1[expert_indices, ...]
    mlp1_bias = w1_bias[expert_indices, ...]
    t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
    t = swiglu(t, limit=7)

    # MLP #2
    mlp2_weight = w2[expert_indices, ...]
    mlp2_bias = w2_bias[expert_indices, ...]
    t = torch.einsum("beck,bek->bec", mlp2_weight, t)
    t += mlp2_bias

    # Weighted sum of experts
    t = torch.einsum("bec,be->bc", t, expert_weights)

    return t


def run_benchmark(num_token, tp,
                  warmup_iter=30, num_iter=200,
                  log=True, profile=False):
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    M = num_token
    E = ModelConfig.num_experts
    K = ModelConfig.hidden_size
    N = ModelConfig.intermediate_size // tp
    topk = ModelConfig.experts_per_token

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=3,
                active=5,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(
                    f"./benchmark/logs/{f.__name__}", f"rank_{dist.get_rank()}"
                )
            ),
        )

    (
        x,
        w1,
        w1_bias,
        w2,
        w2_bias,
        exp_data,
    ) = init_compute_data(M, K, N, E, dtype, device)

    for _ in range(warmup_iter):
        _ = oai_moe_forward(
            hidden_states=x,
            w1=w1,
            w1_bias=w1_bias,
            w2=w2,
            w2_bias=w2_bias,
            gating_output=exp_data,
            topk=topk,
        )

    torch.cuda.synchronize(device=device)

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    for _ in range(num_iter):
        _ = oai_moe_forward(
            hidden_states=x,
            w1=w1,
            w1_bias=w1_bias,
            w2=w2,
            w2_bias=w2_bias,
            gating_output=exp_data,
            topk=topk,
        )
        if profile:
            profiler.start()

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0  # total iteration time

    if profile:
        profiler.stop()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_gb = peak_bytes / (1024**3)

    print(f"{num_iter / time:.6f} iter/s, {time:.3f} sec, {peak_mem_gb:.3f} GiB")
    return time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse model configuration arguments.")

    parser.add_argument("-m", type=int, default=1, help="Number of tokens.")
    parser.add_argument("-tp", type=int, default=8, help="Tensor parallelism")
    parser.add_argument("--num_iter", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--profile", action='store_true', help="Enable profiling.")

    args = parser.parse_args()
    num_token = args.m
    tp = args.tp
    num_iter = args.num_iter
    profile = args.profile

    run_benchmark(num_token, tp, num_iter=num_iter, profile=profile)
