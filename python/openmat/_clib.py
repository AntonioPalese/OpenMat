"""
Loads OpenMat.so and declares the C-ABI signatures.
Import this module; use CLIB as the ctypes handle.

Signatures are declared once per dtype in _declare_dtype(), mirroring the
per-dtype body the C++ side generates from openmat_capi_impl.inc.
"""
import ctypes
import os
import pathlib

from ._dtypes import DTYPES


def _find_lib():
    # 1. Env override
    env = os.environ.get("OPENMAT_LIB")
    if env:
        return env
    here = pathlib.Path(__file__).resolve().parent
    # 2. Installed alongside this module (wheel / pip install)
    bundled = here / "OpenMat.so"
    if bundled.exists():
        return str(bundled)
    # 3. Development layout: <repo>/build/OpenMat.so
    dev = here.parent.parent / "build" / "OpenMat.so"
    if dev.exists():
        return str(dev)
    raise FileNotFoundError(
        "Cannot find OpenMat.so. "
        "Build with ./compile.sh or set OPENMAT_LIB=/path/to/OpenMat.so"
    )


CLIB = ctypes.CDLL(_find_lib())

_sz = ctypes.c_size_t
_i = ctypes.c_int
_p = ctypes.c_void_p
_sp = ctypes.POINTER(_sz)
_cp = ctypes.c_char_p
_v = None
_ERR_LEN = 512


def _errbuf():
    return ctypes.create_string_buffer(_ERR_LEN)


def _check_ptr(ptr, errbuf):
    """Pointer-returning convention: nullptr means failure."""
    if not ptr:
        raise RuntimeError(errbuf.value.decode(errors="replace") or "OpenMat: unknown error")
    return ptr


def _check_int(rc, errbuf):
    """int-returning convention: non-zero means failure."""
    if rc != 0:
        raise RuntimeError(errbuf.value.decode(errors="replace") or "OpenMat: unknown error")


def _declare(name, restype, argtypes):
    fn = getattr(CLIB, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


# ── runtime / stream API (dtype-independent) ─────────────────────────────────
_declare("om_cuda_device_count", _i, [])
_declare("om_cuda_is_available", _i, [])
_declare("om_device_synchronize", _i, [_cp, _i])
_declare("om_stream_create", _p, [_cp, _i])
_declare("om_stream_destroy", _v, [_p])
_declare("om_stream_retain", _v, [_p])
_declare("om_stream_release", _v, [_p])
_declare("om_stream_synchronize", _i, [_p, _cp, _i])
_declare("om_stream_handle", _p, [_p])


def _declare_dtype(dt):
    """Declare every om_tensor_<suffix>_* signature for one dtype."""
    s, C = dt.suffix, dt.ctype
    Cp = ctypes.POINTER(C)

    def d(name, restype, argtypes):
        _declare(f"om_tensor_{s}_{name}", restype, argtypes)

    # lifecycle
    for factory in ("create", "zeros", "ones"):
        d(factory, _p, [_sp, _sz, _i, _cp, _i])
    d("full", _p, [_sp, _sz, C, _i, _cp, _i])
    d("from_buffer", _p, [Cp, _sz, _sp, _sz, _i, _cp, _i])
    d("from_buffer_stream", _p, [Cp, _sz, _sp, _sz, _i, _p, _cp, _i])
    d("destroy", _v, [_p])
    d("copy", _p, [_p, _cp, _i])

    # metadata (infallible)
    d("rank", _sz, [_p])
    d("size", _sz, [_p])
    d("itemsize", _sz, [_p])
    d("shape", _v, [_p, _sp])
    d("stride", _v, [_p, _sp])
    d("on_cuda", _i, [_p])
    d("device_id", _i, [_p])
    d("dtype", _cp, [_p])
    d("data_ptr", _p, [_p])

    # data access
    d("to_host", _i, [_p, Cp, _cp, _i])
    d("to_device", _i, [_p, _p, _cp, _i])
    d("fill", _i, [_p, C, _cp, _i])
    d("get_item", _i, [_p, _sp, _sz, Cp, _cp, _i])
    d("set_item", _i, [_p, _sp, _sz, C, _cp, _i])

    # device transfer
    d("cpu", _p, [_p, _cp, _i])
    d("cuda", _p, [_p, _cp, _i])
    d("cpu_stream", _p, [_p, _p, _cp, _i])
    d("cuda_stream", _p, [_p, _p, _cp, _i])
    d("to", _p, [_p, _cp, _cp, _i])
    d("to_stream", _p, [_p, _cp, _p, _cp, _i])

    # arithmetic — tensor x tensor
    for name in ("add", "sub", "mul", "div", "matmul"):
        d(name, _p, [_p, _p, _cp, _i])
        d(f"{name}_stream", _p, [_p, _p, _p, _cp, _i])

    # arithmetic — tensor x scalar
    for name in ("add", "sub", "mul", "div"):
        d(f"{name}_scalar", _p, [_p, C, _cp, _i])
        d(f"{name}_scalar_stream", _p, [_p, C, _p, _cp, _i])

    # reductions
    for name in ("sum", "mean", "min", "max"):
        d(name, C, [_p, _cp, _i])

    # shape manipulation
    d("reshape", _p, [_p, _sp, _sz, _cp, _i])
    d("flatten", _p, [_p, _cp, _i])
    d("squeeze", _p, [_p, _sz, _cp, _i])
    d("unsqueeze", _p, [_p, _sz, _cp, _i])
    d("transpose", _p, [_p, _cp, _i])
    d("transpose_stream", _p, [_p, _p, _cp, _i])
    d("permute", _p, [_p, _sp, _sz, _cp, _i])
    d("permute_stream", _p, [_p, _sp, _sz, _p, _cp, _i])

    # fused ops
    for name in ("relu", "sigmoid"):
        d(name, _p, [_p, _cp, _i])
        d(f"{name}_stream", _p, [_p, _p, _cp, _i])
    for name in ("scale_shift", "shift_scale"):
        d(name, _p, [_p, C, C, _cp, _i])
    for name in ("fused_add_mul", "fused_sub_mul", "fused_mul_add", "fused_div_add"):
        d(name, _p, [_p, _p, C, _cp, _i])


for _dt in DTYPES:
    _declare_dtype(_dt)
del _dt
