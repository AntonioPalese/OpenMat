"""
DLPack: zero-copy exchange with PyTorch, NumPy, CuPy, JAX.

A DLPack tensor travels as a PyCapsule named "dltensor" wrapping a C
DLManagedTensor (headers/dlpack.h). Whoever consumes the capsule renames it
"used_dltensor" and becomes responsible for calling its deleter; a capsule
that is garbage-collected still named "dltensor" was never consumed, so its
destructor runs the deleter instead. Exactly one of the two happens.

The C side does the tensor work (om_tensor_<dtype>_to_dlpack / _from_dlpack);
this module only moves pointers in and out of capsules, through the CPython
C API that ctypes.pythonapi exposes.
"""
import ctypes

from ._clib import CLIB, _errbuf, _check_ptr, _ERR_LEN
from ._dtypes import float32, int32

# PyCapsule_New keeps the name *pointer*, not a copy, so the names must be
# objects that live as long as the process: module-level bytes constants.
_DLTENSOR = b"dltensor"
_USED_DLTENSOR = b"used_dltensor"

# DLPack device types and dtype codes (headers/dlpack.h).
_kDLCPU, _kDLCUDA, _kDLCUDAHost, _kDLCUDAManaged = 1, 2, 3, 13
_DTYPES = {(2, 32): float32, (0, 32): int32}          # (code, bits) -> DType

_Destructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _Destructor]

_capsule_get = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_get.restype = ctypes.c_void_p
_capsule_get.argtypes = [ctypes.py_object, ctypes.c_char_p]

_capsule_set_name = ctypes.pythonapi.PyCapsule_SetName
_capsule_set_name.restype = ctypes.c_int
_capsule_set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]

# The destructor receives the dying capsule as a raw pointer: taking it as a
# py_object would add a reference to an object whose refcount is already 0.
# So the two calls it makes need prototypes over void* instead.
_capsule_is_valid_raw = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi))
_capsule_get_raw = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi))


def _destroy_unconsumed(capsule):
    try:
        if _capsule_is_valid_raw(capsule, _DLTENSOR):
            CLIB.om_dlpack_delete(_capsule_get_raw(capsule, _DLTENSOR))
    except Exception:   # never raise out of a destructor, not even at shutdown
        pass


# Kept alive for the life of the process: every exported capsule points at it.
_DESTRUCTOR = _Destructor(_destroy_unconsumed)


def export_capsule(tensor) -> object:
    """Wrap `tensor` (an openmat.Tensor) in a "dltensor" capsule."""
    eb = _errbuf()
    ptr = _check_ptr(tensor._fn("to_dlpack")(tensor._h, tensor._stream_h, eb, _ERR_LEN), eb)
    try:
        return _capsule_new(ptr, _DLTENSOR, _DESTRUCTOR)
    except BaseException:
        CLIB.om_dlpack_delete(ptr)
        raise


def from_dlpack(obj):
    """Wrap a DLPack-capable object (or a "dltensor" capsule) without copying.

    The result shares memory with `obj`: writes through either side are
    visible through the other, and the producer's memory stays alive until
    the last OpenMat view of it is gone. Supported element types are float32
    and int32, on CPU or CUDA; anything else raises TypeError and leaves
    `obj` untouched. CUDA tensors are requested on the legacy default stream,
    the stream OpenMat's default-stream operations run on, so the producer
    orders its pending work before ours.
    """
    from .tensor import Tensor

    if type(obj).__name__ == "PyCapsule":
        capsule = obj
    elif hasattr(obj, "__dlpack__"):
        dev_type, _ = obj.__dlpack_device__()
        if dev_type in (_kDLCUDA, _kDLCUDAManaged):
            capsule = obj.__dlpack__(stream=1)
        else:
            capsule = obj.__dlpack__()
    else:
        raise TypeError(f"from_dlpack: {type(obj).__name__} does not support DLPack")

    ptr = _capsule_get(capsule, _DLTENSOR)     # ValueError if already consumed
    code, bits, lanes, dev_type, dev_id = (ctypes.c_int() for _ in range(5))
    CLIB.om_dlpack_info(ptr, ctypes.byref(code), ctypes.byref(bits), ctypes.byref(lanes),
                        ctypes.byref(dev_type), ctypes.byref(dev_id))
    dt = _DTYPES.get((code.value, bits.value)) if lanes.value == 1 else None
    if dt is None:
        raise TypeError(
            f"from_dlpack: unsupported element type (DLPack code {code.value}, "
            f"{bits.value} bits, {lanes.value} lanes); OpenMat has float32 and "
            f"int32 — convert on the producer side first, e.g. x.float()")
    if dev_type.value not in (_kDLCPU, _kDLCUDAHost, _kDLCUDA, _kDLCUDAManaged):
        raise TypeError(f"from_dlpack: unsupported device type {dev_type.value}")

    eb = _errbuf()
    h = _check_ptr(getattr(CLIB, f"om_tensor_{dt.suffix}_from_dlpack")(ptr, eb, _ERR_LEN), eb)
    # The tensor now owns the DLManagedTensor: mark the capsule consumed so
    # its destructor does not run the deleter a second time.
    _capsule_set_name(capsule, _USED_DLTENSOR)
    return Tensor._wrap(h, dt)
