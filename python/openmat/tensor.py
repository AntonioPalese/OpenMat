"""
Pythonic Tensor wrapper around the OpenMat C-ABI.

Usage:
    from openmat import Tensor, Stream

    a = Tensor.zeros([3, 4])
    b = Tensor.ones([3, 4], device="cuda")
    c = (a + b).sum()

    with Stream() as s:                 # synchronizes on exit
        d = a.add(a, stream=s)
"""
import ctypes
from typing import Any, List, Optional, Sequence, Union

from ._clib import CLIB, _errbuf, _check_ptr, _check_int, _ERR_LEN
from ._dtypes import DType, dtype as _as_dtype, float32, int32
from .stream import Stream, _handle_of

_sz = ctypes.c_size_t

Scalar = Union[int, float]


def _shape_array(shape: Sequence[int]):
    shape = tuple(int(s) for s in shape)
    return (_sz * len(shape))(*shape), len(shape)


def _flatten(data) -> List:
    """Depth-first flatten of nested sequences."""
    out = []
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _infer_shape(data) -> List[int]:
    shape = []
    probe = data
    while isinstance(probe, (list, tuple)):
        shape.append(len(probe))
        if not probe:
            break
        probe = probe[0]
    return shape or [1]


def _parse_device(device: Optional[str]) -> str:
    """Normalize a device spec to the "<kind>:<id>" form om::Device parses."""
    if device is None:
        return "cpu:0"
    kind, _, index = str(device).lower().partition(":")
    if kind not in ("cpu", "cuda"):
        raise ValueError(f"Unknown device '{device}'. Use 'cpu', 'cuda' or 'cuda:<id>'.")
    return f"{kind}:{int(index) if index else 0}"


def _device_flag(device: Optional[str]) -> int:
    return 1 if _parse_device(device).startswith("cuda") else 0


def _is_operand(x) -> bool:
    """True for things an operator can combine with a Tensor."""
    return isinstance(x, (Tensor, int, float))


class Tensor:
    """Wraps an om::Tensor<T> via the C-ABI FFI layer."""

    __slots__ = ("_h", "_dt", "_stream_h")

    # ── construction ──────────────────────────────────────────────────────

    def __init__(self, data, shape: Optional[Sequence[int]] = None,
                 dtype: Any = None, device: str = "cpu"):
        """Build a tensor from nested sequences, a flat sequence or a numpy array."""
        if hasattr(data, "__array_interface__") or type(data).__name__ == "ndarray":
            built = Tensor.from_numpy(data, device=device, dtype=dtype)
            if shape is not None:
                built = built.reshape(shape)
        else:
            flat = _flatten(data) if isinstance(data, (list, tuple)) else [data]
            sh = list(shape) if shape is not None else _infer_shape(data)
            built = Tensor.from_list(flat, sh, device=device, dtype=dtype or float32)
        # adopt the freshly built tensor's handle
        self._h, self._dt, self._stream_h = built._h, built._dt, built._stream_h
        built._h = None  # ownership transferred

    @classmethod
    def _wrap(cls, handle, dt: DType, stream: Optional[Stream] = None) -> "Tensor":
        """Internal: adopt an owning handle returned by the C API."""
        t = cls.__new__(cls)
        t._h = handle
        t._dt = dt
        # Memory allocated on a stream must be freed on that same stream, so the
        # tensor takes a reference on it (released in __del__, after the tensor
        # itself is destroyed).  The reference lives in the C layer and the
        # tensor holds only an integer, so no Python finalization order can put
        # the free after the stream's destruction.
        t._stream_h = _handle_of(stream)
        if t._stream_h:
            CLIB.om_stream_retain(t._stream_h)
        return t

    def __del__(self):
        try:
            h = self._h
        except AttributeError:
            return
        if h:
            try:
                getattr(CLIB, f"om_tensor_{self._dt.suffix}_destroy")(h)
            finally:
                self._h = None
                sh, self._stream_h = self._stream_h, None
                if sh:
                    CLIB.om_stream_release(sh)

    def _fn(self, name: str):
        return getattr(CLIB, f"om_tensor_{self._dt.suffix}_{name}")

    def _same_dtype(self, other: "Tensor"):
        if not isinstance(other, Tensor):
            raise TypeError(f"expected a Tensor, got {type(other).__name__}")
        if other._dt != self._dt:
            raise TypeError(
                f"dtype mismatch: {self._dt.name} and {other._dt.name}. "
                f"Convert one with .astype() first."
            )

    def __repr__(self):
        return (f"Tensor(shape={self.shape}, dtype={self._dt.name}, "
                f"device={self.device!r})")

    def __str__(self):
        return f"{self.tolist()}  # {self._dt.name} {self.device}"

    # ── factories ─────────────────────────────────────────────────────────

    @staticmethod
    def _factory(name: str, shape, device, dtype, *extra):
        dt = _as_dtype(dtype) if dtype is not None else float32
        arr, rank = _shape_array(shape)
        eb = _errbuf()
        fn = getattr(CLIB, f"om_tensor_{dt.suffix}_{name}")
        h = fn(arr, rank, *extra, _device_flag(device), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), dt)

    @staticmethod
    def empty(shape: Sequence[int], device: str = "cpu", dtype: Any = None) -> "Tensor":
        """Uninitialized tensor — contents are whatever the allocator handed back."""
        return Tensor._factory("create", shape, device, dtype)

    @staticmethod
    def zeros(shape: Sequence[int], device: str = "cpu", dtype: Any = None) -> "Tensor":
        return Tensor._factory("zeros", shape, device, dtype)

    @staticmethod
    def ones(shape: Sequence[int], device: str = "cpu", dtype: Any = None) -> "Tensor":
        return Tensor._factory("ones", shape, device, dtype)

    @staticmethod
    def full(shape: Sequence[int], value: Scalar,
             device: str = "cpu", dtype: Any = None) -> "Tensor":
        dt = _as_dtype(dtype) if dtype is not None else float32
        return Tensor._factory("full", shape, device, dt, dt.ctype(dt.py_type(value)))

    @staticmethod
    def from_list(data: Sequence[Scalar], shape: Optional[Sequence[int]] = None,
                  device: str = "cpu", dtype: Any = None,
                  stream: Optional[Stream] = None) -> "Tensor":
        dt = _as_dtype(dtype) if dtype is not None else float32
        flat = _flatten(list(data))
        n = len(flat)
        buf = (dt.ctype * n)(*[dt.py_type(v) for v in flat])
        arr, rank = _shape_array(shape if shape is not None else [n])
        eb = _errbuf()
        if stream is None:
            h = getattr(CLIB, f"om_tensor_{dt.suffix}_from_buffer")(
                buf, n, arr, rank, _device_flag(device), eb, _ERR_LEN)
        else:
            # Places the H2D copy on `stream` rather than the default stream; the
            # C layer synchronizes it before returning, so `buf` is safe to drop.
            h = getattr(CLIB, f"om_tensor_{dt.suffix}_from_buffer_stream")(
                buf, n, arr, rank, _device_flag(device), _handle_of(stream), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), dt, stream)

    @staticmethod
    def _dtype_for_numpy(np_dtype) -> DType:
        """Nearest supported dtype for a numpy one — float64 narrows to float32."""
        import numpy as np
        if np.issubdtype(np_dtype, np.floating):
            return float32
        if np.issubdtype(np_dtype, np.integer) or np.issubdtype(np_dtype, np.bool_):
            return int32
        raise TypeError(f"cannot represent numpy dtype {np_dtype} as an OpenMat dtype")

    @staticmethod
    def from_numpy(array, device: str = "cpu", dtype: Any = None) -> "Tensor":
        """Create a Tensor from a numpy ndarray (always copied).

        Without an explicit `dtype`, any float array narrows to float32 and any
        integer array to int32 — those are the only element types OpenMat has.
        """
        import numpy as np
        src = np.asarray(array)
        dt = _as_dtype(dtype) if dtype is not None else Tensor._dtype_for_numpy(src.dtype)
        arr = np.ascontiguousarray(src, dtype=np.dtype(dt.name))
        buf = arr.ctypes.data_as(ctypes.POINTER(dt.ctype))
        sh, rank = _shape_array(arr.shape if arr.ndim else (1,))
        eb = _errbuf()
        h = getattr(CLIB, f"om_tensor_{dt.suffix}_from_buffer")(
            buf, arr.size, sh, rank, _device_flag(device), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), dt)

    @staticmethod
    def arange(stop: Scalar, start: Scalar = 0, step: Scalar = 1,
               device: str = "cpu", dtype: Any = None) -> "Tensor":
        dt = _as_dtype(dtype) if dtype is not None else float32
        values = []
        v = start
        while (step > 0 and v < stop) or (step < 0 and v > stop):
            values.append(v)
            v += step
        return Tensor.from_list(values, [len(values)], device=device, dtype=dt)

    def copy(self) -> "Tensor":
        eb = _errbuf()
        h = self._fn("copy")(self._h, eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt, self.stream)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    # ── metadata ──────────────────────────────────────────────────────────

    @property
    def rank(self) -> int:
        return int(self._fn("rank")(self._h))

    ndim = rank

    @property
    def size(self) -> int:
        return int(self._fn("size")(self._h))

    @property
    def shape(self) -> List[int]:
        buf = (_sz * self.rank)()
        self._fn("shape")(self._h, buf)
        return list(buf)

    @property
    def stride(self) -> List[int]:
        """Strides in elements (not bytes), row-major."""
        buf = (_sz * self.rank)()
        self._fn("stride")(self._h, buf)
        return list(buf)

    @property
    def dtype(self) -> DType:
        return self._dt

    @property
    def itemsize(self) -> int:
        return int(self._fn("itemsize")(self._h))

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    @property
    def is_cuda(self) -> bool:
        return bool(self._fn("on_cuda")(self._h))

    @property
    def device_index(self) -> int:
        return int(self._fn("device_id")(self._h))

    @property
    def device(self) -> str:
        return f"cuda:{self.device_index}" if self.is_cuda else "cpu"

    @property
    def stream(self) -> Stream:
        """The stream this tensor's memory is bound to (the default stream if none)."""
        return Stream._from_handle(self._stream_h)

    def synchronize(self) -> "Tensor":
        """Block until the stream this tensor was produced on has finished."""
        self.stream.synchronize()
        return self

    def data_ptr(self) -> int:
        """Raw address of the buffer (a device pointer for CUDA tensors).

        Borrowed — valid only while this tensor is alive.
        """
        return int(self._fn("data_ptr")(self._h) or 0)

    def __len__(self) -> int:
        sh = self.shape
        if not sh:
            raise TypeError("len() of a 0-d tensor")
        return sh[0]

    # ── data access ───────────────────────────────────────────────────────

    def _host_buffer(self):
        n = self.size
        buf = (self._dt.ctype * n)()
        eb = _errbuf()
        _check_int(self._fn("to_host")(self._h, buf, eb, _ERR_LEN), eb)
        return buf

    def numpy(self):
        """Return a numpy array holding a copy of the tensor data (on host)."""
        import numpy as np
        buf = self._host_buffer()
        return (np.frombuffer(buf, dtype=np.dtype(self._dt.name))
                  .reshape(self.shape).copy())

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            # numpy >= 2 asks for a no-copy conversion this way; host tensors
            # answer it through __array_interface__, device tensors cannot.
            raise ValueError(
                "cannot convert a CUDA tensor to numpy without copying; "
                "use .cpu() first")
        arr = self.numpy()
        return arr if dtype is None else arr.astype(dtype)

    @property
    def __array_interface__(self):
        """Zero-copy numpy view of a host tensor (numpy keeps this tensor alive)."""
        if self.is_cuda:
            raise AttributeError("__array_interface__ is unavailable for CUDA tensors")
        item = self.itemsize
        return {
            "shape": tuple(self.shape),
            "typestr": self._dt.typestr,
            "data": (self.data_ptr(), False),
            "strides": tuple(s * item for s in self.stride),
            "version": 3,
        }

    @property
    def __cuda_array_interface__(self):
        """CUDA array interface, for zero-copy interop with cupy / torch."""
        if not self.is_cuda:
            raise AttributeError("__cuda_array_interface__ is unavailable for host tensors")
        item = self.itemsize
        return {
            "shape": tuple(self.shape),
            "typestr": self._dt.typestr,
            "data": (self.data_ptr(), False),
            "strides": tuple(s * item for s in self.stride),
            "stream": self.stream.handle or None,
            "version": 3,
        }

    def _nest(self, flat, shape):
        if not shape:
            return flat[0]
        if len(shape) == 1:
            return list(flat)
        step = len(flat) // shape[0] if shape[0] else 0
        return [self._nest(flat[i * step:(i + 1) * step], shape[1:])
                for i in range(shape[0])]

    def tolist(self):
        """Nested python lists mirroring the tensor's shape."""
        return self._nest(list(self._host_buffer()), self.shape)

    def flat(self) -> List[Scalar]:
        """The elements in row-major order, as a flat python list."""
        return list(self._host_buffer())

    def item(self) -> Scalar:
        if self.size != 1:
            raise ValueError(f"item(): tensor has {self.size} elements, expected 1")
        return self._host_buffer()[0]

    def copy_to_device(self, dest: "Tensor") -> "Tensor":
        """Copy this host tensor's elements into `dest`'s device buffer.

        Writes through dest's existing allocation instead of producing a new
        tensor, which is what om::Tensor::copyToDevice does.
        """
        self._same_dtype(dest)
        if self.is_cuda:
            raise ValueError("copy_to_device: source is already on the device")
        if not dest.is_cuda:
            raise ValueError("copy_to_device: destination must be a CUDA tensor")
        if dest.size != self.size:
            raise ValueError(
                f"copy_to_device: size mismatch ({self.size} vs {dest.size})")
        eb = _errbuf()
        _check_int(self._fn("to_device")(self._h, dest.data_ptr(), eb, _ERR_LEN), eb)
        return dest

    def fill(self, value: Scalar) -> "Tensor":
        eb = _errbuf()
        _check_int(self._fn("fill")(self._h, self._dt.ctype(self._dt.py_type(value)),
                                    eb, _ERR_LEN), eb)
        return self

    def _index_array(self, key):
        idx = key if isinstance(key, tuple) else (key,)
        rank = self.rank
        if len(idx) != rank:
            raise IndexError(
                f"expected {rank} indices for a rank-{rank} tensor, got {len(idx)}. "
                f"Partial indexing would need a view, which OpenMat does not have yet."
            )
        shape = self.shape
        norm = []
        for axis, i in enumerate(idx):
            if not isinstance(i, int):
                raise TypeError(f"index must be an int, got {type(i).__name__}")
            if i < 0:
                i += shape[axis]
            norm.append(i)
        return (_sz * rank)(*norm), rank

    def __getitem__(self, key) -> Scalar:
        arr, rank = self._index_array(key)
        out = self._dt.ctype()
        eb = _errbuf()
        _check_int(self._fn("get_item")(self._h, arr, rank,
                                        ctypes.byref(out), eb, _ERR_LEN), eb)
        return out.value

    def __setitem__(self, key, value: Scalar):
        arr, rank = self._index_array(key)
        eb = _errbuf()
        _check_int(self._fn("set_item")(self._h, arr, rank,
                                        self._dt.ctype(self._dt.py_type(value)),
                                        eb, _ERR_LEN), eb)

    # ── device transfer ───────────────────────────────────────────────────

    def _transfer(self, name: str, stream: Optional[Stream], *args) -> "Tensor":
        eb = _errbuf()
        if stream is None:
            h = self._fn(name)(self._h, *args, eb, _ERR_LEN)
        else:
            h = self._fn(f"{name}_stream")(self._h, *args, _handle_of(stream), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt, stream)

    def cpu(self, stream: Optional[Stream] = None) -> "Tensor":
        return self._transfer("cpu", stream)

    def cuda(self, stream: Optional[Stream] = None) -> "Tensor":
        return self._transfer("cuda", stream)

    def to(self, device: str, stream: Optional[Stream] = None) -> "Tensor":
        return self._transfer("to", stream, _parse_device(device).encode())

    def astype(self, dtype: Any, copy: bool = True) -> "Tensor":
        """Convert to another element type (host round-trip; no GPU cast kernel yet)."""
        dt = _as_dtype(dtype)
        if dt == self._dt:
            return self.copy() if copy else self
        flat = [dt.py_type(v) for v in self.cpu().flat()] if self.is_cuda \
            else [dt.py_type(v) for v in self.flat()]
        return Tensor.from_list(flat, self.shape, device=self.device, dtype=dt)

    # ── arithmetic ────────────────────────────────────────────────────────

    def _binop_tt(self, other: "Tensor", name: str,
                  stream: Optional[Stream] = None) -> "Tensor":
        self._same_dtype(other)
        eb = _errbuf()
        if stream is None:
            h = self._fn(name)(self._h, other._h, eb, _ERR_LEN)
        else:
            h = self._fn(f"{name}_stream")(self._h, other._h, _handle_of(stream),
                                           eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt, stream)

    def _binop_ts(self, scalar: Scalar, name: str,
                  stream: Optional[Stream] = None) -> "Tensor":
        if not isinstance(scalar, (int, float)):
            raise TypeError(
                f"expected a Tensor or a number, got {type(scalar).__name__}")
        val = self._dt.ctype(self._dt.py_type(scalar))
        eb = _errbuf()
        if stream is None:
            h = self._fn(f"{name}_scalar")(self._h, val, eb, _ERR_LEN)
        else:
            h = self._fn(f"{name}_scalar_stream")(self._h, val, _handle_of(stream),
                                                  eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt, stream)

    def _dispatch(self, other, name, stream=None):
        if isinstance(other, Tensor):
            return self._binop_tt(other, name, stream)
        return self._binop_ts(other, name, stream)

    def add(self, other, stream: Optional[Stream] = None) -> "Tensor":
        return self._dispatch(other, "add", stream)

    def sub(self, other, stream: Optional[Stream] = None) -> "Tensor":
        return self._dispatch(other, "sub", stream)

    def mul(self, other, stream: Optional[Stream] = None) -> "Tensor":
        return self._dispatch(other, "mul", stream)

    def div(self, other, stream: Optional[Stream] = None) -> "Tensor":
        return self._dispatch(other, "div", stream)

    def matmul(self, other: "Tensor", stream: Optional[Stream] = None) -> "Tensor":
        return self._binop_tt(other, "matmul", stream)

    # Operators return NotImplemented for foreign operands so Python can fall
    # back to the other object's reflected method (and raise the standard
    # TypeError if there is none).

    def __add__(self, other):
        return self._dispatch(other, "add") if _is_operand(other) else NotImplemented

    def __sub__(self, other):
        return self._dispatch(other, "sub") if _is_operand(other) else NotImplemented

    def __mul__(self, other):
        return self._dispatch(other, "mul") if _is_operand(other) else NotImplemented

    def __truediv__(self, other):
        return self._dispatch(other, "div") if _is_operand(other) else NotImplemented

    def __matmul__(self, other):
        return self.matmul(other) if isinstance(other, Tensor) else NotImplemented

    def __radd__(self, other):
        return self._binop_ts(other, "add") if _is_operand(other) else NotImplemented

    def __rmul__(self, other):
        return self._binop_ts(other, "mul") if _is_operand(other) else NotImplemented

    def __rsub__(self, other):
        # other - self  ==  (self * -1) + other
        if not _is_operand(other):
            return NotImplemented
        return self._binop_ts(-1, "mul")._binop_ts(other, "add")

    def __rtruediv__(self, other):
        # other / self — no reciprocal kernel, so go through a full tensor
        if not _is_operand(other):
            return NotImplemented
        return Tensor.full(self.shape, other, device=self.device,
                           dtype=self._dt)._binop_tt(self, "div")

    def __neg__(self):
        return self._binop_ts(-1, "mul")

    def __pos__(self):
        return self.copy()

    # ── reductions ────────────────────────────────────────────────────────

    def _reduce(self, name: str) -> Scalar:
        eb = _errbuf()
        return self._fn(name)(self._h, eb, _ERR_LEN)

    def sum(self) -> Scalar: return self._reduce("sum")
    def mean(self) -> Scalar: return self._reduce("mean")
    def min(self) -> Scalar: return self._reduce("min")
    def max(self) -> Scalar: return self._reduce("max")

    # ── shape manipulation ────────────────────────────────────────────────

    def _shape_op(self, name: str, shape: Sequence[int],
                  stream: Optional[Stream] = None) -> "Tensor":
        arr, rank = _shape_array(shape)
        eb = _errbuf()
        if stream is None:
            h = self._fn(name)(self._h, arr, rank, eb, _ERR_LEN)
        else:
            h = self._fn(f"{name}_stream")(self._h, arr, rank, _handle_of(stream),
                                           eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt, stream)

    def reshape(self, *new_shape) -> "Tensor":
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = new_shape[0]
        return self._shape_op("reshape", new_shape)

    def flatten(self) -> "Tensor":
        eb = _errbuf()
        h = self._fn("flatten")(self._h, eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt)

    def squeeze(self, axis: int) -> "Tensor":
        eb = _errbuf()
        h = self._fn("squeeze")(self._h, _sz(axis), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt)

    def unsqueeze(self, axis: int) -> "Tensor":
        eb = _errbuf()
        h = self._fn("unsqueeze")(self._h, _sz(axis), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt)

    def transpose(self, stream: Optional[Stream] = None) -> "Tensor":
        """Rank-2 transpose; use permute() for higher ranks."""
        eb = _errbuf()
        if stream is None:
            h = self._fn("transpose")(self._h, eb, _ERR_LEN)
        else:
            h = self._fn("transpose_stream")(self._h, _handle_of(stream), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt, stream)

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def permute(self, *axes, stream: Optional[Stream] = None) -> "Tensor":
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = axes[0]
        return self._shape_op("permute", axes, stream)

    # ── fused ops ─────────────────────────────────────────────────────────

    def _unary_fused(self, name: str, stream: Optional[Stream] = None) -> "Tensor":
        eb = _errbuf()
        if stream is None:
            h = self._fn(name)(self._h, eb, _ERR_LEN)
        else:
            h = self._fn(f"{name}_stream")(self._h, _handle_of(stream), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt, stream)

    def relu(self, stream: Optional[Stream] = None) -> "Tensor":
        return self._unary_fused("relu", stream)

    def sigmoid(self, stream: Optional[Stream] = None) -> "Tensor":
        return self._unary_fused("sigmoid", stream)

    def _affine(self, name: str, a: Scalar, b: Scalar) -> "Tensor":
        eb = _errbuf()
        cast = lambda v: self._dt.ctype(self._dt.py_type(v))
        h = self._fn(name)(self._h, cast(a), cast(b), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt)

    def scale_shift(self, scale: Scalar, shift: Scalar) -> "Tensor":
        """x * scale + shift, in one kernel."""
        return self._affine("scale_shift", scale, shift)

    def shift_scale(self, shift: Scalar, scale: Scalar) -> "Tensor":
        """(x + shift) * scale, in one kernel."""
        return self._affine("shift_scale", shift, scale)

    def _binary_fused(self, name: str, other: "Tensor", scalar: Scalar) -> "Tensor":
        self._same_dtype(other)
        eb = _errbuf()
        h = self._fn(name)(self._h, other._h,
                           self._dt.ctype(self._dt.py_type(scalar)), eb, _ERR_LEN)
        return Tensor._wrap(_check_ptr(h, eb), self._dt)

    def fused_add_mul(self, other: "Tensor", scale: Scalar) -> "Tensor":
        """(self + other) * scale"""
        return self._binary_fused("fused_add_mul", other, scale)

    def fused_sub_mul(self, other: "Tensor", scale: Scalar) -> "Tensor":
        """(self - other) * scale"""
        return self._binary_fused("fused_sub_mul", other, scale)

    def fused_mul_add(self, other: "Tensor", shift: Scalar) -> "Tensor":
        """(self * other) + shift"""
        return self._binary_fused("fused_mul_add", other, shift)

    def fused_div_add(self, other: "Tensor", shift: Scalar) -> "Tensor":
        """(self / other) + shift"""
        return self._binary_fused("fused_div_add", other, shift)
