from base import Layout
from blackwell_scale import BlackwellMXScaleLayout
from blackwell_value import BlackwellMXValueLayout
from hopper_scale import HopperMXScaleLayout
from hopper_value import HopperMXValueLayout
from cdna4_scale import CDNA4MXScaleLayout
from strided import StridedLayout
from triton.language.target_info import cuda_capability_geq, is_hip_cdna4

__all__ = [
    "Layout",
    "BlackwellMXValueLayout",
    "BlackwellMXScaleLayout",
    "HopperMXScaleLayout",
    "HopperMXValueLayout",
    "CDNA4MXScaleLayout",
    "StridedLayout",
]


def make_default_matmul_mxfp4_w_layout(mx_axis: int):
    if cuda_capability_geq(10):
        return BlackwellMXValueLayout, dict()
    elif cuda_capability_geq(9):
        return HopperMXValueLayout, {"mx_axis": mx_axis}
    else:
        return StridedLayout, dict()


def make_default_matmul_mxfp4_w_scale_layout(mx_axis: int, num_warps: int = 8):
    if is_hip_cdna4():
        return CDNA4MXScaleLayout, dict()
    else:
        if cuda_capability_geq(10):
            return BlackwellMXScaleLayout, dict()
        elif cuda_capability_geq(9):
            return HopperMXScaleLayout, {"mx_axis": mx_axis, "num_warps": num_warps}

    return StridedLayout, dict()
