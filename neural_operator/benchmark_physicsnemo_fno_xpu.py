import argparse
from copy import deepcopy
import torch
from torchtnt.utils.flops import FlopTensorDispatchMode
import neuralop
from physicsnemo.models.fno import FNO

from collections import defaultdict
def get_max_flops(flop_count_dict, max_value = 0):
    for _, value in flop_count_dict.items():
        # if not nested, compare leaf value to max
        if isinstance(value, int):
            max_value = max(max_value, value)

        # otherwise compute recursive max value below node
        elif isinstance(value, defaultdict):
            new_val = get_max_flops(value, max_value)
            max_value = max(max_value, new_val)
    return max_value

def run_benchmark(batch_size, in_chan, out_chan, hidden_chan, n_layers,
                  forward_only=False, warmup_iter=10, num_iter=100,
                  log=True, profile=False):

    device = torch.device(f"xpu:0")
    torch.xpu.set_device(device)

    fno_ref = neuralop.models.FNO(n_modes=(64,64),
                                  in_channels=in_chan,
                                  out_channels=out_chan,
                                  hidden_channels=hidden_chan,
                                  n_layers=n_layers)

    fno = FNO(in_channels=in_chan,
              out_channels=out_chan,
              decoder_layers=2,
              decoder_layer_size=32,
              dimension=2,
              latent_channels=hidden_chan,
              num_fno_modes=(64,64),
              num_fno_layers=n_layers,
              padding=0)

    fno_ref = fno_ref.to(device)
    fno = fno.to(device)

    model_input = torch.randn(batch_size, in_chan, 128, 128, device=device)

    with FlopTensorDispatchMode(fno_ref) as ftdm:
        # count forward flops
        res = fno_ref(model_input).mean()

        fno_forward_flops = deepcopy(ftdm.flop_counts)

        ftdm.reset()
        res.backward()
        fno_backward_flops = deepcopy(ftdm.flop_counts)

        forward_flops = get_max_flops(fno_forward_flops)
        backward_flops = get_max_flops(fno_backward_flops)
        #print(f"Max FLOPS required for FNO.forward: {forward_flops}")
        #print(f"Max FLOPS required for FNO.backward: {backward_flops}")

    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
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

    for _ in range(warmup_iter):
        res = fno(model_input).mean()
        res.backward()

    if profile:
        profiler.start()

    begin = torch.xpu.Event(enable_timing=True)
    begin.record()

    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = fno(model_input).mean()
                if profile:
                    profiler.step()
    else:
        for _ in range(num_iter):
            res = fno(model_input).mean()
            res.backward()
            if profile:
                profiler.step()

    end = torch.xpu.Event(enable_timing=True)
    end.record()
    torch.xpu.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if forward_only:
        TFLOPS = forward_flops/(time/num_iter)/1e12
    else:
        TFLOPS = backward_flops/(time/num_iter)/1e12

    if profile:
        profiler.stop()

    print(f"{num_iter / time:.6f} iter/s, {time:.3f} sec, {TFLOPS:.1f} TFLOPS")
    return TFLOPS

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse model configuration arguments.")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--in_chan", type=int, default=1, help="Number of channels in input function.")
    parser.add_argument("--out_chan", type=int, default=1, help="Number of channels in output function.")
    parser.add_argument("--hidden_chan", type=int, default=64, help="width of the FNO (i.e. number of channels)")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of Fourier Layers, by default 4")
    parser.add_argument("--forward_only", action='store_true', help="Benchmark forward pass only.")
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
    parser.add_argument("--profile", action='store_true', help="Enable profiling.")

    args = parser.parse_args()
    batch_size = args.batch_size
    in_chan = args.in_chan
    out_chan = args.out_chan
    hidden_chan = args.hidden_chan
    n_layers = args.n_layers
    num_iter = args.num_iter
    forward_only = args.forward_only
    profile = args.profile

    torch.xpu.empty_cache()
    run_benchmark(
       batch_size, in_chan, out_chan, hidden_chan, n_layers,
       forward_only, num_iter=num_iter,
       log=True, profile=profile
    )
