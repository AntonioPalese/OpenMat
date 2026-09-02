"""
The rest of the om::Tensor surface exposed through the FFI: metadata,
element access, shape ops, fused ops and the buffer protocols.
"""
import pytest

import openmat as om
from openmat import Tensor

from .conftest import approx, requires_cuda


# ── construction ──────────────────────────────────────────────────────────────

def test_constructor_from_nested_lists():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == [2, 3]
    assert t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_constructor_infers_rank1():
    assert Tensor([1, 2, 3]).shape == [3]


def test_constructor_with_explicit_shape():
    assert Tensor([1, 2, 3, 4], shape=[2, 2]).shape == [2, 2]


def test_empty_has_the_requested_shape():
    assert Tensor.empty([3, 5]).shape == [3, 5]


def test_arange():
    assert Tensor.arange(5).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert Tensor.arange(7, start=1, step=2).tolist() == [1.0, 3.0, 5.0]


def test_module_level_factories():
    assert om.zeros([2, 2]).shape == [2, 2]
    assert approx(om.ones([4]).sum(), 4.0)
    assert approx(om.full([3], 2.0).sum(), 6.0)


# ── metadata ──────────────────────────────────────────────────────────────────

def test_stride_is_row_major(device):
    t = Tensor.zeros([2, 3, 4], device=device)
    assert t.stride == [12, 4, 1]


def test_nbytes_and_itemsize(device):
    t = Tensor.zeros([2, 3], device=device)
    assert t.itemsize == 4
    assert t.nbytes == 24


def test_len_is_the_first_axis():
    assert len(Tensor.zeros([7, 2])) == 7


def test_device_string_roundtrip(device):
    t = Tensor.zeros([2], device=device)
    assert t.device == ("cpu" if device == "cpu" else "cuda:0")
    assert t.device_index == 0


def test_data_ptr_is_non_null(device):
    assert Tensor.ones([4], device=device).data_ptr() != 0


def test_bad_device_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        Tensor.zeros([2], device="tpu")


# ── element access ────────────────────────────────────────────────────────────

def test_getitem(device):
    t = Tensor.from_list([1, 2, 3, 4, 5, 6], [2, 3], device=device)
    assert approx(t[0, 0], 1.0)
    assert approx(t[1, 2], 6.0)


def test_getitem_negative_indices(device):
    t = Tensor.from_list([1, 2, 3, 4], [2, 2], device=device)
    assert approx(t[-1, -1], 4.0)


def test_setitem(device):
    t = Tensor.zeros([2, 2], device=device)
    t[1, 0] = 9.0
    assert approx(t.sum(), 9.0)
    assert approx(t[1, 0], 9.0)


def test_index_out_of_range(device):
    t = Tensor.zeros([2, 2], device=device)
    with pytest.raises(RuntimeError, match="out of range"):
        t[5, 0]


def test_partial_indexing_is_refused():
    t = Tensor.zeros([2, 3])
    with pytest.raises(IndexError, match="expected 2 indices"):
        t[0]


def test_item(device):
    assert approx(Tensor.full([1], 4.0, device=device).item(), 4.0)
    with pytest.raises(ValueError, match="expected 1"):
        Tensor.zeros([3], device=device).item()


def test_tolist_is_nested(device):
    t = Tensor.from_list(list(range(8)), [2, 2, 2], device=device)
    assert t.tolist() == [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]
    assert t.flat() == [float(i) for i in range(8)]


# ── shape manipulation ────────────────────────────────────────────────────────

def test_transpose(device):
    t = Tensor.from_list([1, 2, 3, 4, 5, 6], [2, 3], device=device)
    tr = t.transpose()
    assert tr.shape == [3, 2]
    assert tr.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert t.T.tolist() == tr.tolist()


def test_transpose_requires_rank2(device):
    with pytest.raises(RuntimeError, match="rank-2"):
        Tensor.zeros([2, 3, 4], device=device).transpose()


def test_permute(device):
    t = Tensor.from_list(list(range(24)), [2, 3, 4], device=device)
    p = t.permute(2, 0, 1)
    assert p.shape == [4, 2, 3]
    assert approx(p.sum(), t.sum())
    assert approx(p[3, 1, 2], t[1, 2, 3])


def test_permute_accepts_a_sequence(device):
    assert Tensor.zeros([2, 3], device=device).permute([1, 0]).shape == [3, 2]


def test_permute_rejects_duplicate_axes(device):
    with pytest.raises(RuntimeError, match="duplicate"):
        Tensor.zeros([2, 3], device=device).permute([0, 0])


def test_reshape_varargs():
    assert Tensor.zeros([6]).reshape(2, 3).shape == [2, 3]
    assert Tensor.zeros([6]).reshape([3, 2]).shape == [3, 2]


# ── arithmetic ────────────────────────────────────────────────────────────────

def test_negation_and_reflected_ops(device):
    t = Tensor.from_list([1, 2, 4], [3], device=device)
    assert (-t).tolist() == [-1.0, -2.0, -4.0]
    assert (10 - t).tolist() == [9.0, 8.0, 6.0]
    assert (10 + t).tolist() == [11.0, 12.0, 14.0]
    assert (2 * t).tolist() == [2.0, 4.0, 8.0]
    assert (8 / t).tolist() == [8.0, 4.0, 2.0]


def test_div_tensor(device):
    a = Tensor.from_list([10, 20, 30], [3], device=device)
    b = Tensor.from_list([2, 4, 5], [3], device=device)
    assert (a / b).tolist() == [5.0, 5.0, 6.0]


def test_matmul_shape_mismatch(device):
    a = Tensor.zeros([2, 3], device=device)
    b = Tensor.zeros([2, 3], device=device)
    with pytest.raises(RuntimeError, match="inner dimensions"):
        a @ b


def test_non_tensor_operand_rejected():
    with pytest.raises(TypeError):
        Tensor.ones([2]) + "x"


# ── fused ops ─────────────────────────────────────────────────────────────────

def test_relu(device):
    t = Tensor.from_list([-2, -1, 0, 1, 2], [5], device=device)
    assert t.relu().tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_sigmoid(device):
    t = Tensor.from_list([0.0], [1], device=device)
    assert approx(t.sigmoid().item(), 0.5)


def test_shift_scale(device):
    t = Tensor.from_list([1, 2, 3], [3], device=device)
    # (x + 1) * 2
    assert t.shift_scale(1.0, 2.0).tolist() == [4.0, 6.0, 8.0]


def test_scale_shift(device):
    t = Tensor.from_list([1, 2, 3], [3], device=device)
    # x * 2 + 1
    assert t.scale_shift(2.0, 1.0).tolist() == [3.0, 5.0, 7.0]


def test_binary_fused_ops(device):
    a = Tensor.from_list([4, 6, 8], [3], device=device)
    b = Tensor.from_list([2, 2, 2], [3], device=device)
    assert a.fused_add_mul(b, 2.0).tolist() == [12.0, 16.0, 20.0]
    assert a.fused_sub_mul(b, 2.0).tolist() == [4.0, 8.0, 12.0]
    assert a.fused_mul_add(b, 1.0).tolist() == [9.0, 13.0, 17.0]
    assert a.fused_div_add(b, 1.0).tolist() == [3.0, 4.0, 5.0]


# ── raw buffer copies ─────────────────────────────────────────────────────────

@requires_cuda
def test_copy_to_device_writes_in_place():
    host = Tensor.from_list([1, 2, 3, 4], [2, 2])
    dev = Tensor.zeros([2, 2], device="cuda")
    ptr = dev.data_ptr()
    host.copy_to_device(dev)
    assert dev.data_ptr() == ptr           # same allocation, written through
    assert dev.tolist() == [[1.0, 2.0], [3.0, 4.0]]


@requires_cuda
def test_copy_to_device_checks_operands():
    host = Tensor.ones([4])
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        host.copy_to_device(Tensor.zeros([4]))
    with pytest.raises(ValueError, match="size mismatch"):
        host.copy_to_device(Tensor.zeros([8], device="cuda"))
    with pytest.raises(ValueError, match="already on the device"):
        Tensor.ones([4], device="cuda").copy_to_device(Tensor.zeros([4], device="cuda"))


# ── numpy interop ─────────────────────────────────────────────────────────────

def test_array_interface_is_zero_copy():
    np = pytest.importorskip("numpy")
    t = Tensor.from_list([1, 2, 3, 4], [2, 2])
    a = np.asarray(t)
    assert a.base is t                 # numpy keeps the tensor alive
    t[0, 0] = 99.0
    assert a[0, 0] == 99.0             # and sees writes through it


def test_numpy_copies():
    np = pytest.importorskip("numpy")
    t = Tensor.from_list([1, 2, 3, 4], [2, 2])
    a = t.numpy()
    t[0, 0] = 99.0
    assert a[0, 0] == 1.0


def test_from_numpy_narrows_float64():
    np = pytest.importorskip("numpy")
    t = Tensor.from_numpy(np.array([1.5, 2.5]))     # numpy's default is float64
    assert t.dtype == om.float32
    assert t.tolist() == [1.5, 2.5]


def test_constructor_accepts_a_float64_array():
    np = pytest.importorskip("numpy")
    assert Tensor(np.arange(6.0).reshape(2, 3)).shape == [2, 3]


def test_from_numpy_rejects_complex():
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError, match="cannot represent"):
        Tensor.from_numpy(np.array([1 + 2j]))


def test_from_numpy_preserves_int_dtype():
    np = pytest.importorskip("numpy")
    t = Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.int32))
    assert t.dtype == om.int32
    assert t.tolist() == [[1, 2], [3, 4]]


@requires_cuda
def test_array_interface_unavailable_on_cuda():
    t = Tensor.ones([4], device="cuda")
    assert not hasattr(t, "__array_interface__")
    assert hasattr(t, "__cuda_array_interface__")
    cai = t.__cuda_array_interface__
    assert cai["shape"] == (4,)
    assert cai["typestr"] == "<f4"
    assert cai["data"][0] == t.data_ptr()


def test_cuda_array_interface_unavailable_on_host():
    assert not hasattr(Tensor.ones([4]), "__cuda_array_interface__")
