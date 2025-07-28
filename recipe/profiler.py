'''
https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

Steps
Import all necessary libraries

Instantiate a simple Resnet model

Using profiler to analyze execution time

Using profiler to analyze memory consumption

Using tracing functionality

Examining stack traces

Using profiler to analyze long-running jobs
'''

import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(
    prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=10
    )
)

with profile(
    activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


# Add both CPU and GPU activities
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    device = "cuda"
    activities += [ProfilerActivity.CUDA]
elif torch.xpu.is_available():
    device = "xpu"
    activities += [ProfilerActivity.XPU]
else:
    print(
        "Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices"
    )
    import sys

    sys.exit(0)

sort_by_keyword = device + "_time_total"

model = models.resnet18().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)

with profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

print(
    prof.key_averages(group_by_input_shape=True).table(
        sort_by=sort_by_keyword, row_limit=10
    )
)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

#--------------------------------------------------------------------
sort_by_keyword = "self_" + device + "_time_total"

with profile(
    activities=activities,
    with_stack=True,
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
) as prof:
    model(inputs)

# Print aggregated stats
print(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=2))


from torch.profiler import schedule

#my_schedule = schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=2)

sort_by_keyword = "self_" + device + "_time_total"

'''
At the end of each cycle profiler calls the specified on_trace_ready function and passes itself as an argument. This function is used to process the new trace - either by obtaining the table output or by saving the output on disk as a trace file.

To send the signal to the profiler that the next step has started, call prof.step() function. The current profiler step is stored in prof.step_num.
'''
def trace_handler(p):
    output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    print(output)
    p.export_chrome_trace("trace_" + str(p.step_num) + ".json")


with profile(
    activities=activities,
    schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=3),
    on_trace_ready=trace_handler,
) as p:
    for idx in range(8):
        model(inputs)
        p.step()
