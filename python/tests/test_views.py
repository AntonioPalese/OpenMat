"""Views: indexing, slicing, reshape and friends sharing one storage.

NumPy is the reference throughout: the same key applied to an ndarray with
the same contents must give the same values, and a write through the view
must show up in the base exactly where NumPy's would.
"""
import gc

import numpy as np
import pytest

import openmat as om

from .conftest import requires_cuda


def base(device, shape=(4, 6)):
    n = int(np.prod(shape))
    ref = np.arange(1, n + 1, dtype=np.float32).reshape(shape)
    return om.Tensor(ref, device=device), ref


@pytest.mark.parametrize("key", [
    1, -1, (2,), (slice(1, 3),), (slice(None), 2), (slice(None), slice(1, None, 2)),
    (slice(None, None, 2), slice(0, 5, 3)), (Ellipsis, 3), (1, Ellipsis),
    (slice(3, 1),), (slice(-3, None), slice(None, -2)),
])
def test_getitem_matches_numpy(device, key):
    t, ref = base(device)
    v = t[key]
    assert v.shape == list(ref[key].shape)
    np.testing.assert_array_equal(v.numpy(), ref[key])
    if v.size:
        assert v.shares_memory(t)


def test_full_integer_index_is_still_a_scalar(device):
    t, ref = base(device)
    assert t[2, 3] == ref[2, 3]
    assert t[-1, -1] == ref[-1, -1]


def test_write_through_view_reaches_base(device):
    t, ref = base(device)
    t[:, 1::2] = 0.0
    ref[:, 1::2] = 0.0
    t[1] = 7.0
    ref[1] = 7.0
    np.testing.assert_array_equal(t.numpy(), ref)


def test_setitem_broadcasts_tensors_and_lists(device):
    t = om.Tensor.zeros([3, 4], device=device)
    t[:, 0::2] = om.Tensor([1.0, 2.0], device=device)
    t[2] = [9, 8, 7, 6]
    assert t.tolist() == [[1, 0, 2, 0], [1, 0, 2, 0], [9, 8, 7, 6]]


def test_ops_on_strided_views_match_numpy(device):
    t, ref = base(device, (5, 8))
    even, odd = t[:, 0::2], t[:, 1::2]
    re, ro = ref[:, 0::2], ref[:, 1::2]
    assert not even.is_contiguous()
    np.testing.assert_allclose((even + odd).numpy(), re + ro)
    np.testing.assert_allclose((odd / even).numpy(), ro / re, rtol=1e-6)
    np.testing.assert_allclose((even * 2 - 3).numpy(), re * 2 - 3)
    np.testing.assert_allclose((odd - 20).relu().numpy(), np.maximum(ro - 20, 0))
    assert even.sum() == pytest.approx(re.sum())
    assert odd.max() == ro.max()
    np.testing.assert_allclose((even @ om.Tensor(np.ones((4, 2), np.float32),
                                                 device=device)).numpy(),
                               re @ np.ones((4, 2), np.float32))


def test_inplace_on_view(device):
    t, ref = base(device)
    t[1:3, 2:].add_(10.0)
    ref[1:3, 2:] += 10.0
    col = t[:, 0]
    col *= 2.0
    ref[:, 0] *= 2.0
    np.testing.assert_array_equal(t.numpy(), ref)


def test_reshape_is_a_view_when_contiguous(device):
    t, _ = base(device)
    r = t.reshape(3, 8)
    assert r.shares_memory(t)
    r.fill_(1.0)
    assert t.sum() == 24.0

    strided = t[:, ::2]
    flat = strided.reshape(12)                 # has to copy
    assert not flat.shares_memory(t)
    assert flat.is_contiguous()


def test_squeeze_unsqueeze_are_views(device):
    t = om.Tensor.zeros([2, 1, 3], device=device)
    s = t.squeeze(1)
    u = s.unsqueeze(0)
    assert s.shares_memory(t) and u.shares_memory(t)
    u[0, 1, 2] = 5.0
    assert t[1, 0, 2] == 5.0


def test_view_metadata(device):
    t, _ = base(device)
    v = t[1:, 1::2]
    assert v.stride == [6, 2]
    assert v.storage_offset == 7
    assert not v.is_contiguous()
    c = v.contiguous()
    assert c.is_contiguous() and not c.shares_memory(t)
    assert t.contiguous().shares_memory(t)
    assert not t.clone().shares_memory(t)


def test_numpy_protocols_see_the_view(device):
    t, ref = base(device)
    v = t[1:3, ::2]
    if device == "cpu":
        arr = np.asarray(v)                    # zero-copy via __array_interface__
        np.testing.assert_array_equal(arr, ref[1:3, ::2])
        v[0, 0] = -1.0
        assert arr[0, 0] == -1.0
    else:
        cai = v.__cuda_array_interface__
        assert cai["strides"] == (24, 8)
        assert cai["data"][0] == v.data_ptr()


def test_bad_keys(device):
    t, _ = base(device)
    with pytest.raises(IndexError, match="step must be positive"):
        t[::-1]
    with pytest.raises(IndexError, match="too many indices"):
        t[0, 0, 0]
    with pytest.raises(IndexError, match="single ellipsis"):
        t[..., ...]
    with pytest.raises(TypeError):
        t["a"]


def test_iteration_yields_rows(device):
    t, ref = base(device, (3, 2))
    rows = [r.tolist() for r in t]
    assert rows == ref.tolist()


def test_view_outlives_base(device):
    t, _ = base(device)
    v = t[2]
    del t
    gc.collect()
    assert v.tolist() == [13, 14, 15, 16, 17, 18]


@requires_cuda
def test_view_keeps_its_stream_alive():
    # The view's storage came from a stream's pool and must be freed there,
    # after the Stream object and the base tensor are gone.
    s = om.Stream()
    a = om.Tensor.ones([64, 64], device="cuda")
    r = a.add(a, stream=s)
    v = r[10:12]
    s.synchronize()
    del r, s
    gc.collect()
    assert v.sum() == 2.0 * 128
    del v
    gc.collect()
