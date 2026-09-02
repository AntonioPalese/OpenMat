"""
OpenMat — Python bindings.

A ctypes binding over the C-ABI layer compiled into OpenMat.so.  The surface
mirrors om::Tensor<T>: factories, arithmetic, reductions, shape manipulation,
fused ops, device transfer, and the CUDA streams that all of them run on.
"""
from ._dtypes import DType, dtype, float32, int32
from .stream import Stream, cuda_is_available, device_count, synchronize
from .tensor import Tensor

__all__ = [
    "Tensor",
    "Stream",
    "DType",
    "dtype",
    "float32",
    "int32",
    "cuda_is_available",
    "device_count",
    "synchronize",
    "zeros",
    "ones",
    "full",
    "empty",
    "arange",
    "from_list",
    "from_numpy",
]

__version__ = "0.2.0"

# Module-level aliases for the factories — openmat.zeros([2,3]) reads better
# than openmat.Tensor.zeros([2,3]).
zeros = Tensor.zeros
ones = Tensor.ones
full = Tensor.full
empty = Tensor.empty
arange = Tensor.arange
from_list = Tensor.from_list
from_numpy = Tensor.from_numpy
