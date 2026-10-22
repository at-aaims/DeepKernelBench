#
# Copyright IBM Corp. 2024
# SPDX-License-Identifier: MIT
#
"""Capacity-aware buffer allocation for the communication benchmarks.

The sweeps in this directory run to message sizes that not every device can
hold. A device that runs out part way through should skip the sizes it cannot
fit and still report the ones it measured, rather than losing the whole sweep
to an uncaught allocation failure.

The skip has to be unanimous. These are collective benchmarks: if one rank
gives up on a size and its peers go on to call all_reduce for it, the peers
block until something kills the job. A hang is a worse failure than the
exception it replaced, and a harder one to diagnose. So the decision is not
made locally -- every rank allocates, reports whether it succeeded, and the
ranks agree on a single answer before any of them proceeds.
"""

import sys

import torch
import torch.distributed as dist


def _is_out_of_memory(exc):
    """True if exc is a device capacity failure rather than a real error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    # HIP, and older CUDA builds, can surface capacity failures as a plain
    # RuntimeError, so fall back to matching the message.
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


def allocate_or_skip(*sizes):
    """Allocate one float32 tensor per entry in sizes, on every rank or none.

    Returns a tuple of tensors, or None if any rank could not allocate them,
    in which case the caller should skip this size. Anything that is not an
    out-of-memory failure propagates -- this is meant to sidestep a device
    limit, not to swallow bugs.
    """
    tensors = []
    allocated = 1
    try:
        for nelem in sizes:
            tensors.append(torch.rand(nelem, device='cuda'))
    except Exception as exc:
        if not _is_out_of_memory(exc):
            raise
        allocated = 0

    # One element is small enough to allocate even on a rank that has just run
    # out. MIN means a single failure anywhere skips the size everywhere, so
    # all ranks leave this function with the same verdict.
    verdict = torch.tensor([allocated], device='cuda', dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    if verdict.item() == 1:
        return tuple(tensors)

    # Ranks that did allocate have to let go too, or the next size inherits
    # the pressure that caused this one to be skipped.
    del tensors[:]
    torch.cuda.empty_cache()
    return None


def report_skip(nMB, rank):
    """Record a skipped size on the same stream as the results table.

    Skipped sizes are announced rather than quietly dropped: a sweep that
    stops early otherwise reads as a sweep that covered everything.
    """
    if rank == 0:
        print("{:8.2f}".format(nMB),
              "   skipped: device cannot allocate buffers for this size",
              file=sys.stderr)
