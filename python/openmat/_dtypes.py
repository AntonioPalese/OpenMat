"""
dtype registry.

Each OpenMat dtype maps to (a) the token used in the exported C symbol names
(``om_tensor_<suffix>_*``) and (b) the ctypes scalar used for values crossing
the boundary.  Adding a dtype here plus an inclusion of openmat_capi_impl.inc
on the C++ side is all it takes to widen the FFI.
"""
import ctypes


class DType:
    """An OpenMat element type."""

    __slots__ = ("name", "suffix", "ctype", "itemsize", "py_type")

    def __init__(self, name, suffix, ctype, py_type):
        self.name = name
        self.suffix = suffix
        self.ctype = ctype
        self.itemsize = ctypes.sizeof(ctype)
        self.py_type = py_type

    # numpy-style single-character/typestr used by the buffer protocols
    @property
    def typestr(self):
        kind = "f" if self.py_type is float else "i"
        return f"<{kind}{self.itemsize}"

    def __repr__(self):
        return f"dtype('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        try:
            return self.name == dtype(other).name
        except (TypeError, ValueError):
            return NotImplemented

    def __hash__(self):
        return hash(self.name)


float32 = DType("float32", "float", ctypes.c_float, float)
int32 = DType("int32", "int", ctypes.c_int, int)

DTYPES = (float32, int32)

_ALIASES = {
    "float32": float32, "float": float32, "f32": float32, "f4": float32,
    "int32": int32, "int": int32, "i32": int32, "i4": int32,
    float: float32,
    int: int32,
}


def dtype(spec):
    """Normalize a dtype spec (DType, string, python type, numpy dtype) to a DType."""
    if isinstance(spec, DType):
        return spec
    key = spec
    if not isinstance(spec, (str, type)):
        # numpy dtype / numpy scalar type — fall back to its name
        key = getattr(spec, "name", None) or getattr(spec, "__name__", None)
    try:
        return _ALIASES[key]
    except (KeyError, TypeError):
        supported = ", ".join(d.name for d in DTYPES)
        raise TypeError(f"Unsupported dtype {spec!r}. Supported: {supported}")
