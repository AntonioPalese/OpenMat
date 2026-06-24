"""
Loads OpenMat.so and declares the C-ABI signatures.
Import this module; use CLIB as the ctypes handle.
"""
import ctypes
import os
import pathlib

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

_sz  = ctypes.c_size_t
_f   = ctypes.c_float
_i   = ctypes.c_int
_p   = ctypes.c_void_p
_fp  = ctypes.POINTER(ctypes.c_float)
_sp  = ctypes.POINTER(_sz)
_cp  = ctypes.c_char_p
_ERR_LEN = 512

def _errbuf():
    return ctypes.create_string_buffer(_ERR_LEN)

def _check_ptr(ptr, errbuf):
    if not ptr:
        raise RuntimeError(errbuf.value.decode(errors="replace"))
    return ptr

def _check_int(rc, errbuf):
    if rc != 0:
        raise RuntimeError(errbuf.value.decode(errors="replace"))

# ── lifecycle ────────────────────────────────────────────────────────────────
_sig = CLIB.om_tensor_float_create
_sig.restype = _p
_sig.argtypes = [_sp, _sz, _i, _cp, _i]

_sig = CLIB.om_tensor_float_zeros
_sig.restype = _p
_sig.argtypes = [_sp, _sz, _i, _cp, _i]

_sig = CLIB.om_tensor_float_ones
_sig.restype = _p
_sig.argtypes = [_sp, _sz, _i, _cp, _i]

_sig = CLIB.om_tensor_float_full
_sig.restype = _p
_sig.argtypes = [_sp, _sz, _f, _i, _cp, _i]

_sig = CLIB.om_tensor_float_from_buffer
_sig.restype = _p
_sig.argtypes = [_fp, _sz, _sp, _sz, _i, _cp, _i]

_sig = CLIB.om_tensor_float_destroy
_sig.restype = None
_sig.argtypes = [_p]

_sig = CLIB.om_tensor_float_copy
_sig.restype = _p
_sig.argtypes = [_p, _cp, _i]

# ── metadata ─────────────────────────────────────────────────────────────────
_sig = CLIB.om_tensor_float_rank
_sig.restype = _sz
_sig.argtypes = [_p]

_sig = CLIB.om_tensor_float_size
_sig.restype = _sz
_sig.argtypes = [_p]

_sig = CLIB.om_tensor_float_shape
_sig.restype = None
_sig.argtypes = [_p, _sp]

_sig = CLIB.om_tensor_float_on_cuda
_sig.restype = _i
_sig.argtypes = [_p]

# ── data access ───────────────────────────────────────────────────────────────
_sig = CLIB.om_tensor_float_to_host
_sig.restype = _i
_sig.argtypes = [_p, _fp, _cp, _i]

_sig = CLIB.om_tensor_float_fill
_sig.restype = None
_sig.argtypes = [_p, _f]

# ── device transfer ───────────────────────────────────────────────────────────
_sig = CLIB.om_tensor_float_cpu
_sig.restype = _p
_sig.argtypes = [_p, _cp, _i]

_sig = CLIB.om_tensor_float_cuda
_sig.restype = _p
_sig.argtypes = [_p, _cp, _i]

# ── tensor × tensor arithmetic ────────────────────────────────────────────────
for _name in ("add", "sub", "mul", "div", "matmul"):
    _sig = getattr(CLIB, f"om_tensor_float_{_name}")
    _sig.restype = _p
    _sig.argtypes = [_p, _p, _cp, _i]

# ── tensor × scalar arithmetic ────────────────────────────────────────────────
for _name in ("add_scalar", "sub_scalar", "mul_scalar", "div_scalar"):
    _sig = getattr(CLIB, f"om_tensor_float_{_name}")
    _sig.restype = _p
    _sig.argtypes = [_p, _f, _cp, _i]

# ── reductions ────────────────────────────────────────────────────────────────
for _name in ("sum", "mean", "min", "max"):
    _sig = getattr(CLIB, f"om_tensor_float_{_name}")
    _sig.restype = _f
    _sig.argtypes = [_p, _cp, _i]

# ── shape manipulation ────────────────────────────────────────────────────────
_sig = CLIB.om_tensor_float_reshape
_sig.restype = _p
_sig.argtypes = [_p, _sp, _sz, _cp, _i]

_sig = CLIB.om_tensor_float_flatten
_sig.restype = _p
_sig.argtypes = [_p, _cp, _i]

_sig = CLIB.om_tensor_float_squeeze
_sig.restype = _p
_sig.argtypes = [_p, _sz, _cp, _i]

_sig = CLIB.om_tensor_float_unsqueeze
_sig.restype = _p
_sig.argtypes = [_p, _sz, _cp, _i]

# ── fused ops ─────────────────────────────────────────────────────────────────
_sig = CLIB.om_tensor_float_scale_shift
_sig.restype = _p
_sig.argtypes = [_p, _f, _f, _cp, _i]

_sig = CLIB.om_tensor_float_fused_add_mul
_sig.restype = _p
_sig.argtypes = [_p, _p, _f, _cp, _i]

_sig = CLIB.om_tensor_float_fused_mul_add
_sig.restype = _p
_sig.argtypes = [_p, _p, _f, _cp, _i]
