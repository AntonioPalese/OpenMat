"""NumPy-style broadcasting for the binary ops, checked against NumPy itself.

The shape table mirrors tests/test_broadcast.cpp: right- and left-side bias,
both operands broadcast at once, a size-1 middle axis, and rank 5/6 (the
_nd kernel on the GPU).
"""
import operator

import numpy as np
import pytest

import openmat as om

CASES = [
    ((32, 128), (128,)),
    ((128,), (32, 128)),
    ((4, 1), (1, 5)),
    ((5,), (1,)),
    ((2, 3, 4), (3, 1)),
    ((3, 1, 5), (3, 4, 5)),
    ((2, 3, 1, 4, 5), (3, 4, 1)),
    ((2, 1, 3, 2, 1, 4), (1, 2, 3, 1, 5, 1)),
]

OPS = [operator.add, operator.sub, operator.mul, operator.truediv]


def arange(shape, base, dtype):
    n = int(np.prod(shape))
    return (base + np.arange(n) % 7).astype(dtype).reshape(shape)


@pytest.mark.parametrize("sa,sb", CASES)
@pytest.mark.parametrize("op", OPS, ids=lambda f: f.__name__)
def test_binary_matches_numpy(device, sa, sb, op):
    na, nb = arange(sa, 1, np.float32), arange(sb, 2, np.float32)
    a = om.Tensor.from_numpy(na, device=device)
    b = om.Tensor.from_numpy(nb, device=device)
    got = op(a, b)
    want = op(na, nb)
    assert got.shape == list(want.shape)
    np.testing.assert_allclose(got.cpu().numpy() if got.is_cuda else got.numpy(),
                               want, rtol=1e-6)


def test_int_bias(device):
    na, nb = arange((6, 10), 1, np.int32), arange((10,), 2, np.int32)
    a = om.Tensor.from_numpy(na, device=device)
    b = om.Tensor.from_numpy(nb, device=device)
    got = a * b
    np.testing.assert_array_equal(got.cpu().numpy() if got.is_cuda else got.numpy(),
                                  na * nb)


def test_inplace_bias_keeps_buffer(device):
    nx, nb = arange((32, 128), 1, np.float32), arange((128,), 2, np.float32)
    x = om.Tensor.from_numpy(nx, device=device)
    b = om.Tensor.from_numpy(nb, device=device)
    p = x.data_ptr()
    x.add_(b)
    x += b
    assert x.data_ptr() == p
    np.testing.assert_allclose(x.cpu().numpy() if x.is_cuda else x.numpy(),
                               nx + 2 * nb)


def test_inplace_into_smaller_operand_raises(device):
    x = om.Tensor.zeros([32, 128], device=device)
    b = om.Tensor.zeros([128], device=device)
    with pytest.raises(RuntimeError, match="result shape"):
        b.add_(x)


def test_fused_add_mul_broadcasts(device):
    nx, nb = arange((8, 3, 5), 1, np.float32), arange((3, 1), 2, np.float32)
    x = om.Tensor.from_numpy(nx, device=device)
    b = om.Tensor.from_numpy(nb, device=device)
    got = x.fused_add_mul(b, 2.0)
    np.testing.assert_allclose(got.cpu().numpy() if got.is_cuda else got.numpy(),
                               (nx + nb) * 2.0)


@pytest.mark.parametrize("sa,sb", [((4,), (5,)), ((2, 3), (3, 2))])
def test_incompatible_shapes_raise(device, sa, sb):
    a = om.Tensor.zeros(list(sa), device=device)
    b = om.Tensor.zeros(list(sb), device=device)
    with pytest.raises(RuntimeError, match="not broadcastable"):
        a + b
