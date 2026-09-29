"""DLPack: zero-copy exchange with NumPy and PyTorch.

"Zero-copy" is asserted, not assumed: every round trip checks that both
sides report the same data pointer and that a write through one side is
visible through the other. Ownership is asserted too: the producer's deleter
must run exactly once, when the last OpenMat view is gone, and never for a
capsule that was refused.
"""
import gc
import sys

import numpy as np
import pytest

import openmat as om

from .conftest import requires_cuda

try:
    import torch
except ImportError:           # the NumPy half of this file still runs
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


# ── NumPy (CPU) ─────────────────────────────────────────────────────────────

def test_numpy_to_openmat_shares_memory():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = om.from_dlpack(a)
    assert t.shape == [3, 4] and t.dtype == om.float32
    assert t.data_ptr() == a.ctypes.data
    t[1, 2] = -5.0
    assert a[1, 2] == -5.0
    a[0, 0] = 99.0
    assert t[0, 0] == 99.0


def test_openmat_to_numpy_shares_memory():
    t = om.Tensor.from_list(list(range(6)), [2, 3], dtype=om.int32)
    a = np.from_dlpack(t)
    assert a.dtype == np.int32
    assert a.ctypes.data == t.data_ptr()
    # NumPy marks arrays from an unversioned ("dltensor") capsule read-only:
    # that format cannot say whether writing is allowed. Writes go the other way.
    t[1, 1] = 77
    assert a[1, 1] == 77


def test_strided_views_cross_both_ways():
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    t = om.from_dlpack(a[:, 1::2])                 # foreign strides in
    assert t.stride == [6, 2] and not t.is_contiguous()
    np.testing.assert_array_equal((t * 2).numpy(), a[:, 1::2] * 2)

    base = om.Tensor(np.arange(24, dtype=np.float32).reshape(4, 6))
    b = np.from_dlpack(base[1:, ::3])              # OpenMat strides out
    np.testing.assert_array_equal(b, base.numpy()[1:, ::3])


def test_deleter_runs_once_when_last_view_dies():
    a = np.ones((8,), dtype=np.float32)
    before = sys.getrefcount(a)
    t = om.from_dlpack(a)
    v = t[2:5]
    del t
    gc.collect()
    assert sys.getrefcount(a) > before              # v still holds the export
    del v
    gc.collect()
    assert sys.getrefcount(a) == before


def test_exported_tensor_outlives_openmat():
    t = om.Tensor.from_list([1.0, 2.0, 3.0], [3])
    a = np.from_dlpack(t)
    del t
    gc.collect()
    assert a.tolist() == [1.0, 2.0, 3.0]


def test_unsupported_dtype_is_refused_and_left_intact():
    a = np.zeros(4, dtype=np.float64)
    before = sys.getrefcount(a)
    with pytest.raises(TypeError, match="unsupported element type"):
        om.from_dlpack(a)
    gc.collect()
    assert sys.getrefcount(a) == before            # capsule's own destructor ran


def test_negative_strides_are_refused():
    a = np.arange(4, dtype=np.float32)[::-1]
    with pytest.raises(RuntimeError, match="negative strides"):
        om.from_dlpack(a)


def test_a_consumed_capsule_cannot_be_reused():
    cap = om.Tensor.ones([2]).__dlpack__()
    om.from_dlpack(cap)
    with pytest.raises(ValueError):
        om.from_dlpack(cap)


def test_unconsumed_export_is_released():
    t = om.Tensor.ones([4])
    cap = t.__dlpack__()
    del cap                                        # destructor runs the deleter
    gc.collect()
    assert t.sum() == 4.0


def test_dlpack_device():
    assert om.Tensor.ones([1]).__dlpack_device__() == (1, 0)


# ── PyTorch ─────────────────────────────────────────────────────────────────

@requires_torch
def test_torch_round_trip(device):
    x = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    t = om.from_dlpack(x)
    assert t.data_ptr() == x.data_ptr()
    assert t.device.startswith(device)
    y = torch.from_dlpack(t)
    assert y.data_ptr() == x.data_ptr()
    t.add_(1.0)
    if device == "cuda":
        om.synchronize()
    assert x[0, 0].item() == 1.0 and y[2, 3].item() == 12.0


@requires_torch
def test_torch_non_contiguous_inputs(device):
    x = torch.arange(48, dtype=torch.float32, device=device).reshape(6, 8)
    for view in (x.t(), x[:, ::2], x[1:5, 3:]):
        t = om.from_dlpack(view)
        assert t.stride == list(view.stride())
        got = torch.from_dlpack(t * 3.0)
        torch.testing.assert_close(got, view * 3.0)


@requires_torch
def test_torch_int32(device):
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=device)
    t = om.from_dlpack(x)
    assert t.dtype == om.int32
    assert torch.from_dlpack(t + 10).tolist() == [[11, 12], [13, 14]]


@requires_torch
def test_torch_unsupported_dtype(device):
    x = torch.zeros(3, dtype=torch.float64, device=device)
    with pytest.raises(TypeError, match="float32 and int32"):
        om.from_dlpack(x)
    assert x.sum().item() == 0.0                   # still intact


@requires_torch
@requires_cuda
def test_fused_kernel_inside_torch_code():
    # The use case: an OpenMat fused kernel on PyTorch tensors, no copies.
    x = torch.randn(512, 256, device="cuda")
    b = torch.randn(256, device="cuda")
    out = om.from_dlpack(x).fused_add_mul(om.from_dlpack(b), 2.5)
    y = torch.from_dlpack(out)
    torch.testing.assert_close(y, (x + b) * 2.5)


@requires_torch
@requires_cuda
def test_torch_tensor_outlives_openmat_stream():
    # Exported from a tensor allocated on an OpenMat stream: the storage frees
    # on that stream, so the export must keep the stream alive after every
    # OpenMat object — the Stream included — is gone.
    s = om.Stream()
    a = om.Tensor.ones([256, 256], device="cuda")
    r = a.add(a, stream=s)
    y = torch.from_dlpack(r)
    del r, a, s
    gc.collect()
    assert y.sum().item() == 2.0 * 256 * 256
    del y
    gc.collect()
    torch.cuda.synchronize()


@requires_torch
@requires_cuda
def test_torch_side_stream_consumer():
    # A consumer on a non-default torch stream: the export synchronizes, so
    # the values OpenMat wrote are there when torch reads them on its stream.
    t = om.Tensor.zeros([1 << 20], device="cuda")
    t.add_(3.0)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        y = torch.from_dlpack(t)
        total = y.sum()
    side.synchronize()
    assert total.item() == 3.0 * (1 << 20)
