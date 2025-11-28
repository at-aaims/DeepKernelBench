import torch
import triton
import triton.language as tl
from triton.runtime import driver

def is_xpu():
    return driver.active.get_current_target().backend == "xpu"

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
                   BLOCK_SIZE_X: tl.constexpr,
                   BLOCK_SIZE_Y: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0) * BLOCK_SIZE_Y
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE_X)
    row_offsets = tl.arange(0, BLOCK_SIZE_Y)
    offsets = col_offsets[None, :] + row_offsets[:, None] * input_row_stride
    input_ptrs = row_start_ptr + offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    mask = col_offsets[None, :] < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=1)[:, None]
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

DEVICE = driver.active.get_active_torch_device()

if is_xpu():
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    MAX_WORK_GROUP_SIZE = properties["max_work_group_size"]
else:
    MAX_WORK_GROUP_SIZE = 1024


def softmax(x):
    n_rows, n_cols = x.shape

    y = torch.empty_like(x)

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE_X = triton.next_power_of_2(n_cols)
    BLOCK_SIZE_Y = MAX_WORK_GROUP_SIZE // BLOCK_SIZE_X
    BLOCK_SIZE_Y = BLOCK_SIZE_Y if BLOCK_SIZE_Y > 0 else 1

    # Create a number of persistent programs.
    softmax_kernel[(n_rows // BLOCK_SIZE_Y, )](y, x, x.stride(0), y.stride(0), n_cols,
                                               BLOCK_SIZE_X=BLOCK_SIZE_X, BLOCK_SIZE_Y=BLOCK_SIZE_Y)
    return y

