"""dtype plumbing: the FFI exports one symbol family per element type."""
import pytest

import openmat as om
from openmat import Tensor

from .conftest import approx


def test_dtype_normalization():
    assert om.dtype("float32") is om.float32
    assert om.dtype("f32") is om.float32
    assert om.dtype(float) is om.float32
    assert om.dtype("int32") is om.int32
    assert om.dtype(int) is om.int32
    assert om.dtype(om.int32) is om.int32


def test_unsupported_dtype_rejected():
    with pytest.raises(TypeError, match="Unsupported dtype"):
        om.dtype("float64")


def test_default_dtype_is_float32():
    assert Tensor.zeros([2]).dtype == om.float32
    assert Tensor.zeros([2]).itemsize == 4


def test_int_tensor_roundtrip(device):
    t = Tensor.from_list([1, 2, 3, 4], [2, 2], device=device, dtype="int32")
    assert t.dtype == om.int32
    assert t.tolist() == [[1, 2], [3, 4]]
    assert t.sum() == 10
    assert isinstance(t.sum(), int)


def test_int_arithmetic(device):
    a = Tensor.from_list([1, 2, 3], [3], device=device, dtype=om.int32)
    b = Tensor.from_list([10, 20, 30], [3], device=device, dtype=om.int32)
    assert (a + b).tolist() == [11, 22, 33]
    assert (a * 3).tolist() == [3, 6, 9]
    assert (b - a).tolist() == [9, 18, 27]


def test_int_matmul(device):
    a = Tensor.from_list([1, 2, 3, 4], [2, 2], device=device, dtype="int32")
    eye = Tensor.from_list([1, 0, 0, 1], [2, 2], device=device, dtype="int32")
    assert (a @ eye).tolist() == [[1, 2], [3, 4]]


def test_int_division_truncates(device):
    a = Tensor.from_list([7, 8, 9], [3], device=device, dtype="int32")
    assert (a / 2).tolist() == [3, 4, 4]


def test_int_division_by_zero_yields_zero(device):
    # Integer division by zero is UB in C++ and CUDA, so the library picks a
    # policy explicitly and applies it identically on CPU and GPU: x / 0 == 0,
    # as NumPy's integer division does.
    a = Tensor.from_list([7, -8, 0], [3], device=device, dtype="int32")
    z = Tensor.from_list([0, 0, 0], [3], device=device, dtype="int32")
    assert (a / z).tolist() == [0, 0, 0]
    assert (a / 0).tolist() == [0, 0, 0]


def test_dtype_mismatch_rejected():
    f = Tensor.ones([3])
    i = Tensor.ones([3], dtype="int32")
    with pytest.raises(TypeError, match="dtype mismatch"):
        f + i


def test_astype_roundtrip(device):
    f = Tensor.from_list([1.7, -2.3, 3.9], [3], device=device)
    i = f.astype("int32")
    assert i.dtype == om.int32
    assert i.device == f.device
    assert i.tolist() == [1, -2, 3]
    assert i.astype("float32").tolist() == [1.0, -2.0, 3.0]


def test_astype_same_dtype_copies():
    a = Tensor.from_list([1.0, 2.0], [2])
    b = a.astype("float32")
    b.fill(0.0)
    assert approx(a.sum(), 3.0)


def test_dtype_reported_by_library_matches():
    # the C side owns the canonical name; make sure Python agrees
    from openmat._clib import CLIB
    assert CLIB.om_tensor_float_dtype(Tensor.ones([1])._h) == b"float32"
    assert CLIB.om_tensor_int_dtype(Tensor.ones([1], dtype="int32")._h) == b"int32"
