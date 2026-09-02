"""
CUDA streams — the canonical execution path.

Every stream overload enqueues work and returns immediately; the caller
synchronizes before reading.  Result tensors keep a reference to the Stream
that produced them because stream-ordered memory must be freed on the same
stream it was allocated on.
"""
import gc

import pytest

import openmat as om
from openmat import Stream, Tensor

from .conftest import approx, requires_cuda

pytestmark = requires_cuda


def test_default_stream_is_shared():
    assert Stream.default() is Stream.default()
    assert Stream.default().is_default
    assert Stream.default().handle == 0


def test_created_stream_has_a_handle():
    s = Stream()
    assert not s.is_default
    assert s.handle != 0
    s.close()


def test_close_is_idempotent():
    s = Stream()
    s.close()
    s.close()
    assert s.is_default  # the facade degrades to the default stream once closed


def test_close_defers_while_a_tensor_still_needs_the_stream():
    a = Tensor.ones([1024], device="cuda")
    s = Stream()
    c = a.add(a, stream=s)
    handle = s.handle
    s.close()                     # early close must not destroy the stream under c
    assert s.is_default           # the facade is detached ...
    assert c.stream.handle == handle   # ... but c still holds the real stream
    c.synchronize()
    assert approx(c.sum(), 2048.0)


def test_chained_ops_on_one_stream():
    a = Tensor.from_list([1, 2, 3, 4], [2, 2], device="cuda")
    b = Tensor.ones([2, 2], device="cuda")
    s = Stream()
    c = a.add(b, stream=s)          # [[2,3],[4,5]]
    d = c.mul(2.0, stream=s)        # [[4,6],[8,10]]
    e = d.transpose(stream=s)       # [[4,8],[6,10]]
    s.synchronize()
    assert e.tolist() == [[4.0, 8.0], [6.0, 10.0]]


def test_result_keeps_its_stream_alive():
    a = Tensor.ones([16], device="cuda")
    s = Stream()
    c = a.add(a, stream=s)
    assert c.stream.handle == s.handle
    del s
    gc.collect()
    # the stream is still alive through c; freeing c on it must not fault
    assert approx(c.synchronize().sum(), 32.0)
    del c
    gc.collect()


def test_context_manager_synchronizes_on_exit():
    a = Tensor.full([256], 2.0, device="cuda")
    with Stream() as s:
        b = a.mul(a, stream=s)
    assert approx(b.sum(), 1024.0)


def test_context_manager_synchronizes_on_exception():
    a = Tensor.ones([8], device="cuda")
    with pytest.raises(ValueError):
        with Stream() as s:
            a.add(a, stream=s)
            raise ValueError("boom")


def test_independent_streams():
    a = Tensor.full([1024], 3.0, device="cuda")
    b = Tensor.full([1024], 4.0, device="cuda")
    s1, s2 = Stream(), Stream()
    x = a.add(a, stream=s1)
    y = b.mul(b, stream=s2)
    s1.synchronize()
    s2.synchronize()
    assert approx(x.sum(), 6.0 * 1024)
    assert approx(y.sum(), 16.0 * 1024)


def test_scalar_ops_on_stream():
    a = Tensor.from_list([1, 2, 3], [3], device="cuda")
    with Stream() as s:
        b = a.add(10.0, stream=s).div(2.0, stream=s)
    assert b.tolist() == [5.5, 6.0, 6.5]


def test_transfer_on_stream():
    host = Tensor.from_list([1, 2, 3, 4], [4])
    with Stream() as s:
        dev = host.cuda(stream=s)
    assert dev.is_cuda
    with Stream() as s:
        back = dev.cpu(stream=s)
    assert not back.is_cuda
    assert back.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_to_on_stream():
    host = Tensor.ones([4])
    with Stream() as s:
        dev = host.to("cuda:0", stream=s)
    assert dev.device == "cuda:0"


def test_from_list_on_stream():
    with Stream() as s:
        t = Tensor.from_list([1, 2, 3, 4], [2, 2], device="cuda", stream=s)
    assert t.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_fused_and_permute_on_stream():
    a = Tensor.from_list([x / 8.0 for x in range(-12, 12)], [2, 3, 4], device="cuda")
    with Stream() as s:
        p = a.permute([2, 0, 1], stream=s)
        r = a.relu(stream=s)
        g = a.sigmoid(stream=s)
    assert p.shape == [4, 2, 3]
    assert approx(r.sum(), sum(x / 8.0 for x in range(0, 12)))
    assert 0.0 < g.min() <= g.max() < 1.0


def test_matmul_on_stream():
    a = Tensor.from_list([1, 2, 3, 4], [2, 2], device="cuda")
    with Stream() as s:
        c = a.matmul(a, stream=s)
    assert c.tolist() == [[7.0, 10.0], [15.0, 22.0]]


def test_device_synchronize():
    a = Tensor.ones([64], device="cuda")
    s = Stream()
    a.add(a, stream=s)
    om.synchronize()
    s.close()


def test_stream_type_is_checked():
    a = Tensor.ones([4], device="cuda")
    with pytest.raises(TypeError, match="openmat.Stream"):
        a.add(a, stream="not a stream")


def test_survives_cyclic_collection():
    """Regression: a Stream and its tensors caught in one reference cycle.

    Frames captured in tracebacks put both in a cycle, where finalization order
    is arbitrary — freeing tensor memory on an already-destroyed stream faults.
    """
    for _ in range(50):
        cycle = {}
        s = Stream()
        a = Tensor.ones([4096], device="cuda")
        cycle["tensors"] = [a.add(a, stream=s).mul(3.0, stream=s) for _ in range(4)]
        cycle["stream"] = s
        cycle["self"] = cycle          # the cycle the collector has to break
        s.synchronize()
        del s, a, cycle
        gc.collect()
    om.synchronize()


def test_device_count_positive():
    assert om.device_count() >= 1
