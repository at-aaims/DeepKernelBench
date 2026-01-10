# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Optional
import pandas as pd
import torch
import torch.utils.benchmark as benchmark

def scaled_mm_supported_device():
    if torch.cuda.is_available():
        if torch.version.hip:
            supported_architectures = ['gfx94', 'gfx95']
            gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
            return any(arch in gcn_arch for arch in supported_architectures)
        else:
            return torch.cuda.get_device_capability() >= (9, 0) or torch.cuda.get_device_capability() == (8, 9)
    return False

def get_name_to_moe_shapes_iter(
    shape_gen_name: str,
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    E: Optional[int] = None,
):
    if shape_gen_name == "llama4_17bx8e":
        # num_experts=8, dim=5120
        names_to_shapes = {
            # M, K, N, E
            "moe.experts.w1": (16640, 5120, 8192, 8),
            "moe.experts.w2": (16640, 8192, 5120, 8),
        }
        return names_to_shapes.items()
    elif shape_gen_name == "llama4_17bx16e":
        # num_experts=16, dim=5120
        names_to_shapes = {
            # M, K, N, E
            "moe.experts.w1": (16640, 5120, 8192, 16),
            "moe.experts.w2": (16640, 8192, 5120, 16),
        }
        return names_to_shapes.items()
    elif shape_gen_name == "llama4_17bx64e":
        # num_experts=64, dim=5120
        names_to_shapes = {
            # M, K, N, E
            "moe.experts.w1": (16640, 5120, 5120*2, 64),
            "moe.experts.w2": (16640, 5120*2, 5120, 64),
        }
        return names_to_shapes.items()
    elif shape_gen_name == "custom":
        assert M is not None and K is not None and N is not None and E is not None, (
            "M, K, N, E must be specified for custom shape_gen"
        )
        name_to_shapes = {
            1: (M, K, N, E),
        }
        return name_to_shapes.items()

    raise AssertionError(f"unknown shape_gen_name {shape_gen_name}")


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(30):
        f(*args, **kwargs)

    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.blocked_autorange(min_run_time=1)
    return measurement.mean


def do_benchmarks(
    f,
    *args,
    **kwargs,
):
    # e2e time including kernel launch overhead
    time_sec = benchmark_fn_in_sec(f, *args, **kwargs)
    return time_sec


@torch.inference_mode()
def run(
    n_limit: Optional[int] = None,
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    E: Optional[int] = None,  # dim 0 of B tensor (num experts)
    out_filename: Optional[str] = None,
    shape_gen_name="llama4_17bx16e",
    recipe: str = "rowwise",
):
    device = torch.device(f"cuda:0")
    torch.cuda.set_device(device)

    assert recipe in ("rowwise",), "unsupported"

    headers = (
        "name",
        "recipe",
        "M",
        "K",
        "N",
        "E",
        "ref_time_s",
        "fp8_time_s",
        "fp8_speedup",
    )
    results = []

    dtype = torch.bfloat16
    name_to_shapes = get_name_to_moe_shapes_iter(shape_gen_name, M, K, N, E)

    for idx, (name, (M, K, N, E)) in enumerate(name_to_shapes,):
        if n_limit is not None and idx >= n_limit:
            break
        assert M % E == 0, (
            "tokens (M) must be evenly divisible by num experts (E) for this benchmark"
        )
        tops = 2 * M * N * K * E
        print("M, K, N, E:", M, K, N, E, f"tops: {tops:.2E}")
 
        try:
            # Run bf16 torch._grouped_mm baseline.
            A = torch.randn(M, K, device=device, dtype=dtype)
            B = torch.randn(E, K, N, device=device, dtype=dtype)
            offs = generate_jagged_offs(E, M).to(device)

            ref_time_sec = do_benchmarks(
                torch._grouped_mm,
                A,
                B,
                offs,
            )
            print(
                f"{dtype} time_sec {ref_time_sec:.2E}"
            )
            del A
            del B

            # Run scaled_grouped_mm.
            A_hp = torch.randn(M, K, device=device)
            B_hp_t = (
                torch.randn(E, K, N, device=device)
                .transpose(-2, -1)
                .contiguous()
                .transpose(-2, -1)
            )

            if recipe == "rowwise":
                # TODO: add e5m2
                A = A_hp.to(torch.float8_e4m3fn)
                del A_hp
                B = B_hp_t.to(torch.float8_e4m3fn)
                del B_hp_t
                scale_a = torch.ones(M, device=device)
                scale_b = torch.ones(E, N, device=device)
            else:
                assert False, f"unknown recipe {recipe}"

            def do_scaled_grouped_mm(A, B):
                nonlocal scale_a
                nonlocal scale_b
                nonlocal offs
                return torch._scaled_grouped_mm(A, B, scale_a, scale_b, offs=offs)

            if recipe == "rowwise":
                do_matmul = do_scaled_grouped_mm
            else:
                raise ValueError(f"unknown recipe {recipe}")

            time_sec = do_benchmarks(do_matmul, A, B)
            print(
                f"torch.float8_e4m3 time_sec {time_sec:.2E}"
            )

            del A, B
            if scale_a is not None:
                del scale_a
            if scale_b is not None:
                del scale_b

            results.append(
                [
                    name,
                    recipe,
                    M,
                    K,
                    N,
                    E,
                    ref_time_sec,
                    time_sec,
                    ref_time_sec / time_sec,
                ]
            )
        except Exception as e:
            print('--------------------------------------------------------------------------------')
            print('Exceptions raised during grouped GEMM benchmark:'                                )
            print(e)
            print('--------------------------------------------------------------------------------')

    data_df = pd.DataFrame(results, columns=headers)
    print(data_df)

    if out_filename is not None:
        data_df.to_csv(out_filename)


def generate_jagged_offs(E, M, dtype=torch.int32):
    """
    Generates a tensor of length E, containing random values divisible by 16,
    from 0 to M, in sorted order, and where the final value in the tensor is always M.
    Args:
        E (int): The length of the tensor.
        M (int): The maximum value in the tensor.
    Returns:
        torch.Tensor: A tensor of length E with the specified properties.
    """
    # Ensure M is divisible by 16
    if M % 16 != 0:
        raise ValueError("M must be divisible by 16")

    # Generate a list of possible values
    possible_values = [i for i in range(0, M + 1, 16)]

    # If E is larger than the number of possible values, raise an error
    if E > len(possible_values):
        raise ValueError("E cannot be larger than the number of possible values")

    # Randomly select E - 1 values from the possible values (excluding M)
    selected_values = torch.tensor(random.sample(possible_values[:-1], E - 1))

    # Append M to the selected values
    selected_values = torch.cat((selected_values, torch.tensor([M])))

    # Sort the selected values
    selected_values, _ = torch.sort(selected_values)

    return selected_values.to(dtype)


def main() -> None:
    if not scaled_mm_supported_device():
        print("FP8 is only supported on H100+ and sm_89 and MI300+ devices. Skip the benchmark.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")

    device_name = gpu_name.replace(' ', '_')

    run(shape_gen_name="llama4_17bx8e", out_filename=f"{device_name}_grouped_mm_llama4_17bx8e.txt")
    run(shape_gen_name="llama4_17bx16e", out_filename=f"{device_name}_grouped_mm_llama4_17bx16e.txt")
    run(shape_gen_name="llama4_17bx64e", out_filename=f"{device_name}_grouped_mm_llama4_17bx64e.txt")


if __name__ == "__main__":
    main()
