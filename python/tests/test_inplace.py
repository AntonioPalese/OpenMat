"""In-place ops (add_, mul_, relu_, fill_) and the destination-provided family.

Every test asserts two things: the values are right, and `data_ptr()` did not
move. The second is the point of the feature — an implementation that quietly
allocated a fresh result and rebound the handle would satisfy every value
assertion here.
"""
import pytest

import openmat as om

from .conftest import requires_cuda


def flat(t):
    return t.cpu().flat() if t.is_cuda else t.flat()


# ── in-place ────────────────────────────────────────────────────────────────

def test_inplace_tensor_ops(device):
    a = om.Tensor([1.0, 2.0, 3.0, 4.0], device=device)
    b = om.Tensor.full([4], 2.0, device=device)
    p = a.data_ptr()

    assert flat(a.add_(b)) == [3.0, 4.0, 5.0, 6.0]
    assert flat(a.sub_(b)) == [1.0, 2.0, 3.0, 4.0]
    assert flat(a.mul_(b)) == [2.0, 4.0, 6.0, 8.0]
    assert flat(a.div_(b)) == [1.0, 2.0, 3.0, 4.0]

    assert a.data_ptr() == p
    assert flat(b) == [2.0, 2.0, 2.0, 2.0], "rhs was modified"


def test_inplace_scalar_ops(device):
    a = om.Tensor([1.0, 2.0, 3.0, 4.0], device=device)
    p = a.data_ptr()

    a.add_(1.0).mul_(2.0).sub_(2.0).div_(2.0)

    assert flat(a) == [1.0, 2.0, 3.0, 4.0]
    assert a.data_ptr() == p


def test_inplace_returns_self(device):
    a = om.Tensor([1.0, 2.0], device=device)
    assert a.add_(1.0) is a
    assert a.relu_() is a
    assert a.fill_(0.0) is a


def test_augmented_assignment_does_not_rebind(device):
    a = om.Tensor([1.0, 2.0, 3.0, 4.0], device=device)
    before = a
    p = a.data_ptr()

    a += 1.0
    a *= 2.0
    a -= 2.0
    a /= 2.0

    assert a is before, "+= rebound the name to a new tensor"
    assert a.data_ptr() == p
    assert flat(a) == [1.0, 2.0, 3.0, 4.0]


def test_inplace_unary_and_fill(device):
    a = om.Tensor([-2.0, -1.0, 0.0, 3.0], device=device)
    p = a.data_ptr()

    assert flat(a.relu_()) == [0.0, 0.0, 0.0, 3.0]
    assert flat(a.fill_(5.0)) == [5.0, 5.0, 5.0, 5.0]

    a.sigmoid_()
    assert all(0.0 < v < 1.0 for v in flat(a))
    assert a.data_ptr() == p


def test_inplace_matches_out_of_place(device):
    a = om.Tensor([[-1.0, 2.0], [3.0, -4.0]], device=device)
    b = om.Tensor([[0.5, 1.5], [2.5, 3.5]], device=device)

    want = flat((a * b).relu())
    a.mul_(b).relu_()
    assert flat(a) == want


def test_inplace_self_operand(device):
    a = om.Tensor([1.0, 2.0, 3.0], device=device)
    a.add_(a)
    assert flat(a) == [2.0, 4.0, 6.0]


def test_inplace_int_dtype(device):
    a = om.Tensor([1, 2, 3, 4], dtype=om.int32, device=device)
    a.mul_(3)
    assert flat(a) == [3, 6, 9, 12]


def test_inplace_shape_mismatch_raises(device):
    a = om.Tensor.zeros([4], device=device)
    b = om.Tensor.zeros([5], device=device)
    with pytest.raises(RuntimeError):
        a.add_(b)


def test_inplace_dtype_mismatch_raises():
    a = om.Tensor.zeros([4], dtype=om.float32)
    b = om.Tensor.zeros([4], dtype=om.int32)
    with pytest.raises(TypeError):
        a.add_(b)


# ── destination-provided ────────────────────────────────────────────────────

def test_out_reuses_one_destination(device):
    a = om.Tensor([1.0, 2.0, 3.0, 4.0], device=device)
    b = om.Tensor.full([4], 2.0, device=device)
    out = om.Tensor.zeros([4], device=device)
    p = out.data_ptr()

    assert flat(a.add_out(b, out)) == [3.0, 4.0, 5.0, 6.0]
    assert flat(a.mul_out(b, out)) == [2.0, 4.0, 6.0, 8.0]
    assert flat(a.sub_out(1.0, out)) == [0.0, 1.0, 2.0, 3.0]
    assert flat(a.relu_out(out)) == [1.0, 2.0, 3.0, 4.0]

    assert out.data_ptr() == p
    assert flat(a) == [1.0, 2.0, 3.0, 4.0], "operand was clobbered"


def test_out_returns_the_destination(device):
    a = om.Tensor([1.0, 2.0], device=device)
    out = om.Tensor.zeros([2], device=device)
    assert a.add_out(1.0, out) is out


def test_out_matmul_transpose_permute(device):
    a = om.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    b = om.Tensor.full([3, 2], 1.0, device=device)

    mm = om.Tensor.zeros([2, 2], device=device)
    assert flat(a.matmul_out(b, mm)) == flat(a @ b)

    tr = om.Tensor.zeros([3, 2], device=device)
    assert flat(a.transpose_out(tr)) == flat(a.T)

    pm = om.Tensor.zeros([3, 2], device=device)
    assert flat(a.permute_out([1, 0], pm)) == flat(a.permute(1, 0))


def test_out_wrong_shape_raises(device):
    a = om.Tensor.zeros([4], device=device)
    out = om.Tensor.zeros([5], device=device)
    with pytest.raises(RuntimeError):
        a.add_out(1.0, out)


@requires_cuda
def test_out_wrong_device_raises():
    a = om.Tensor.zeros([4], device="cuda")
    out = om.Tensor.zeros([4], device="cpu")
    with pytest.raises(RuntimeError):
        a.add_out(1.0, out)


def test_out_aliased_destination_rejected_for_matmul(device):
    a = om.Tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    with pytest.raises(RuntimeError):
        a.matmul_out(a, a)
    with pytest.raises(RuntimeError):
        a.transpose_out(a)


# ── streams ─────────────────────────────────────────────────────────────────

@requires_cuda
def test_inplace_on_a_stream():
    a = om.Tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    b = om.Tensor.full([4], 2.0, device="cuda")
    p = a.data_ptr()

    with om.Stream() as s:
        a.mul_(b, stream=s).add_(1.0, stream=s).relu_(stream=s)

    assert a.cpu().flat() == [3.0, 5.0, 7.0, 9.0]
    assert a.data_ptr() == p


@requires_cuda
def test_out_on_a_stream():
    a = om.Tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    b = om.Tensor.full([4], 2.0, device="cuda")
    out = om.Tensor.zeros([4], device="cuda")
    p = out.data_ptr()

    with om.Stream() as s:
        a.add_out(b, out, stream=s)

    assert out.cpu().flat() == [3.0, 4.0, 5.0, 6.0]
    assert out.data_ptr() == p


@requires_cuda
def test_fill_on_a_stream():
    a = om.Tensor.zeros([8], device="cuda")
    with om.Stream() as s:
        a.fill_(3.0, stream=s)
    assert a.cpu().flat() == [3.0] * 8


# ── the reason the feature exists ───────────────────────────────────────────

def test_loop_keeps_one_buffer(device):
    w = om.Tensor.zeros([256], device=device)
    g = om.Tensor.full([256], 0.5, device=device)
    p = w.data_ptr()

    for _ in range(50):
        w.add_(g)
        assert w.data_ptr() == p

    assert flat(w)[0] == pytest.approx(25.0)
