"""
Pythonic Tensor wrapper around the OpenMat C-ABI.

Usage:
    from openmat import Tensor
    a = Tensor.zeros([3, 4])
    b = Tensor.ones([3, 4], device="cuda")
    c = (a + b).sum()
"""
import ctypes
from typing import List, Sequence, Union

from ._clib import CLIB, _errbuf, _check_ptr, _check_int, _ERR_LEN

_sz = ctypes.c_size_t
_f  = ctypes.c_float


def _shape_array(shape: Sequence[int]):
    arr = (_sz * len(shape))(*shape)
    return arr, len(shape)


class Tensor:
    """Wraps an om::Tensor<float> via the C-ABI FFI layer."""

    # ── construction / destruction ─────────────────────────────────────────

    def __init__(self, _handle=None):
        """Internal: pass an opaque handle obtained from CLIB."""
        if _handle is None:
            raise ValueError("Use factory methods: Tensor.zeros(), Tensor.from_numpy(), etc.")
        self._h = _handle  # void* (int in ctypes)

    def __del__(self):
        if getattr(self, "_h", None):
            CLIB.om_tensor_float_destroy(self._h)
            self._h = None

    def __repr__(self):
        return (f"Tensor(shape={self.shape}, device="
                f"{'cuda' if self.is_cuda else 'cpu'})")

    # ── factories ─────────────────────────────────────────────────────────

    @staticmethod
    def _device_flag(device: str) -> int:
        d = device.lower()
        if d.startswith("cuda"):
            return 1
        if d.startswith("cpu"):
            return 0
        raise ValueError(f"Unknown device '{device}'. Use 'cpu' or 'cuda'.")

    @staticmethod
    def zeros(shape: Sequence[int], device: str = "cpu") -> "Tensor":
        arr, rank = _shape_array(shape)
        eb = _errbuf()
        h = CLIB.om_tensor_float_zeros(arr, rank, Tensor._device_flag(device), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    @staticmethod
    def ones(shape: Sequence[int], device: str = "cpu") -> "Tensor":
        arr, rank = _shape_array(shape)
        eb = _errbuf()
        h = CLIB.om_tensor_float_ones(arr, rank, Tensor._device_flag(device), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    @staticmethod
    def full(shape: Sequence[int], value: float, device: str = "cpu") -> "Tensor":
        arr, rank = _shape_array(shape)
        eb = _errbuf()
        h = CLIB.om_tensor_float_full(arr, rank, _f(value), Tensor._device_flag(device), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    @staticmethod
    def from_list(data: List[float], shape: Sequence[int], device: str = "cpu") -> "Tensor":
        n = len(data)
        buf = (ctypes.c_float * n)(*data)
        arr, rank = _shape_array(shape)
        eb = _errbuf()
        h = CLIB.om_tensor_float_from_buffer(buf, n, arr, rank,
                                              Tensor._device_flag(device), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    @staticmethod
    def from_numpy(array, device: str = "cpu") -> "Tensor":
        """Create a Tensor from a numpy ndarray (float32)."""
        import numpy as np
        arr = np.ascontiguousarray(array, dtype=np.float32)
        n = arr.size
        buf = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sh, rank = _shape_array(arr.shape)
        eb = _errbuf()
        h = CLIB.om_tensor_float_from_buffer(buf, n, sh, rank,
                                              Tensor._device_flag(device), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    # ── metadata ──────────────────────────────────────────────────────────

    @property
    def rank(self) -> int:
        return int(CLIB.om_tensor_float_rank(self._h))

    @property
    def size(self) -> int:
        return int(CLIB.om_tensor_float_size(self._h))

    @property
    def shape(self) -> List[int]:
        r = self.rank
        buf = (_sz * r)()
        CLIB.om_tensor_float_shape(self._h, buf)
        return list(buf)

    @property
    def is_cuda(self) -> bool:
        return bool(CLIB.om_tensor_float_on_cuda(self._h))

    @property
    def device(self) -> str:
        return "cuda" if self.is_cuda else "cpu"

    # ── data access ───────────────────────────────────────────────────────

    def numpy(self):
        """Return a float32 numpy array with a copy of the tensor data (on host)."""
        import numpy as np
        n = self.size
        buf = (ctypes.c_float * n)()
        eb = _errbuf()
        _check_int(CLIB.om_tensor_float_to_host(self._h, buf, eb, _ERR_LEN), eb)
        return np.frombuffer(buf, dtype=np.float32).reshape(self.shape).copy()

    def tolist(self) -> List[float]:
        n = self.size
        buf = (ctypes.c_float * n)()
        eb = _errbuf()
        _check_int(CLIB.om_tensor_float_to_host(self._h, buf, eb, _ERR_LEN), eb)
        return list(buf)

    def fill(self, value: float) -> "Tensor":
        CLIB.om_tensor_float_fill(self._h, _f(value))
        return self

    def copy(self) -> "Tensor":
        eb = _errbuf()
        h = CLIB.om_tensor_float_copy(self._h, eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    # ── device transfer ───────────────────────────────────────────────────

    def cpu(self) -> "Tensor":
        eb = _errbuf()
        h = CLIB.om_tensor_float_cpu(self._h, eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    def cuda(self) -> "Tensor":
        eb = _errbuf()
        h = CLIB.om_tensor_float_cuda(self._h, eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    def to(self, device: str) -> "Tensor":
        return self.cuda() if Tensor._device_flag(device) else self.cpu()

    # ── arithmetic operators ──────────────────────────────────────────────

    def _binop_tt(self, other: "Tensor", fn_name: str) -> "Tensor":
        eb = _errbuf()
        fn = getattr(CLIB, f"om_tensor_float_{fn_name}")
        h = fn(self._h, other._h, eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    def _binop_ts(self, scalar: float, fn_name: str) -> "Tensor":
        eb = _errbuf()
        fn = getattr(CLIB, f"om_tensor_float_{fn_name}_scalar")
        h = fn(self._h, _f(scalar), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    def __add__(self, other):
        if isinstance(other, Tensor): return self._binop_tt(other, "add")
        return self._binop_ts(float(other), "add")

    def __sub__(self, other):
        if isinstance(other, Tensor): return self._binop_tt(other, "sub")
        return self._binop_ts(float(other), "sub")

    def __mul__(self, other):
        if isinstance(other, Tensor): return self._binop_tt(other, "mul")
        return self._binop_ts(float(other), "mul")

    def __truediv__(self, other):
        if isinstance(other, Tensor): return self._binop_tt(other, "div")
        return self._binop_ts(float(other), "div")

    def __radd__(self, other): return self._binop_ts(float(other), "add")
    def __rsub__(self, other):
        # other - self → (-self) + other
        return self._binop_ts(-1.0, "mul")._binop_ts(float(other), "add")
    def __rmul__(self, other): return self._binop_ts(float(other), "mul")

    def matmul(self, other: "Tensor") -> "Tensor":
        return self._binop_tt(other, "matmul")

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    # ── reductions ────────────────────────────────────────────────────────

    def _reduce(self, name: str) -> float:
        eb = _errbuf()
        fn = getattr(CLIB, f"om_tensor_float_{name}")
        return float(fn(self._h, eb, _ERR_LEN))

    def sum(self)  -> float: return self._reduce("sum")
    def mean(self) -> float: return self._reduce("mean")
    def min(self)  -> float: return self._reduce("min")
    def max(self)  -> float: return self._reduce("max")

    # ── shape manipulation ────────────────────────────────────────────────

    def reshape(self, new_shape: Sequence[int]) -> "Tensor":
        arr, rank = _shape_array(new_shape)
        eb = _errbuf()
        h = CLIB.om_tensor_float_reshape(self._h, arr, rank, eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    def flatten(self) -> "Tensor":
        eb = _errbuf()
        h = CLIB.om_tensor_float_flatten(self._h, eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    def squeeze(self, axis: int) -> "Tensor":
        eb = _errbuf()
        h = CLIB.om_tensor_float_squeeze(self._h, _sz(axis), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    def unsqueeze(self, axis: int) -> "Tensor":
        eb = _errbuf()
        h = CLIB.om_tensor_float_unsqueeze(self._h, _sz(axis), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    # ── fused ops ─────────────────────────────────────────────────────────

    def scale_shift(self, scale: float, shift: float) -> "Tensor":
        eb = _errbuf()
        h = CLIB.om_tensor_float_scale_shift(self._h, _f(scale), _f(shift), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    def fused_add_mul(self, other: "Tensor", scale: float) -> "Tensor":
        eb = _errbuf()
        h = CLIB.om_tensor_float_fused_add_mul(self._h, other._h, _f(scale), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))

    def fused_mul_add(self, other: "Tensor", shift: float) -> "Tensor":
        eb = _errbuf()
        h = CLIB.om_tensor_float_fused_mul_add(self._h, other._h, _f(shift), eb, _ERR_LEN)
        return Tensor(_check_ptr(h, eb))
