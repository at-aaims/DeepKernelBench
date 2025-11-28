import torch
import triton
import triton.language as tl
from triton.runtime import driver

def is_hip():
    return driver.active.get_current_target().backend == "hip"

def is_xpu():
    return driver.active.get_current_target().backend == "xpu"

def is_cdna():
    return is_hip() and driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908')

DEVICE = driver.active.get_active_torch_device()

properties = driver.active.utils.get_device_properties(DEVICE.index)


NUM_SM = properties["multiprocessor_count"]
SIZE_SMEM = properties["max_shared_mem"]

if is_xpu():
    WARPS_PER_EU = 8  # TODO: Get from properties
    EU_PER_SM = 8  # TODO: Get from properties
    MAX_NUM_WG = 64  # TODO: Get from properties
    WARP_SIZE = properties["sub_group_sizes"][-1]
    WG_SIZE = properties["max_work_group_size"]
    max_num_warps = WG_SIZE // WARP_SIZE
    warps_per_sm = WARPS_PER_EU * EU_PER_SM
    # Possible SLM allocation sizes in kB
    tg_slm_sizes = [i * 2**10 for i in [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]]  # TODO: Get from properties
else:
    NUM_REGS = properties["max_num_regs"]
    WARP_SIZE = properties["warpSize"]

target = driver.active.get_current_target()

# the addition of n_rows
@triton.jit
def softmax2_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)



def softmax2(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if is_xpu():
        # Simple heuristic depending on `BLOCK_SIZE`. We aim for 4 elements per thread as the block size may be almost twice
        # as larger as the row size. This way, we reduce the number of threads performing no work.
        # As the maximum number of warps is limited by hardware, we need to make sure we do not surpass that limit.
        # You will see in the next tutorial how to auto-tune this value in a more natural
        # way so you don't have to come up with manual heuristics yourself.
        num_warps = min(max_num_warps, max(1, BLOCK_SIZE // (WARP_SIZE * 4)))
    else:
        num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax2_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared

    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        num_programs = min(num_programs, n_rows)
    elif is_xpu():
        num_warps = min(max_num_warps, max(1, BLOCK_SIZE // (WARP_SIZE * 4)))

        def allocated_slm_size(size_smem):
            for size in tg_slm_sizes:
                if size_smem <= size:
                    return size
            raise RuntimeError("Exceeded max SLM allocation size")

        num_wg_threads = warps_per_sm // num_warps
        num_wg_slm = MAX_NUM_WG if size_smem == 0 else SIZE_SMEM // allocated_slm_size(size_smem)
        num_wg = min(num_wg_threads, num_wg_slm, MAX_NUM_WG)
        num_programs = NUM_SM * num_wg
        # We will *not* launch a persistent kernel if the number of rows is lower (not needed) or that would imply each
        # program would need to process more than 2 rows. Persistent kernels save thread dispatch overhead, but cannot
        # hide stalling. Overdispatching will help hiding this thanks to work-group level preemption. That's why, as a
        # heuristic, if each work-group would need to process at least more than 2 rows, we do not schedule a persistent
        # kernel.
        if n_rows < num_programs or n_rows // num_programs > 2:
            num_programs = n_rows

    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        num_programs = min(num_programs, n_rows)


    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y

