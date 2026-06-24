import pytest
from openmat import Tensor


# ── helpers ───────────────────────────────────────────────────────────────────

def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


# ── factories ─────────────────────────────────────────────────────────────────

def test_zeros_shape():
    t = Tensor.zeros([3, 4])
    assert t.shape == [3, 4]
    assert t.sum() == 0.0

def test_ones_sum():
    assert Tensor.ones([2, 5]).sum() == 10.0

def test_full():
    assert Tensor.full([4], 3.0).sum() == 12.0

def test_from_list():
    t = Tensor.from_list([1, 2, 3, 4, 5], [5])
    assert t.shape == [5]
    assert t.sum() == 15.0

def test_from_numpy():
    np = pytest.importorskip("numpy")
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = Tensor.from_numpy(arr)
    assert t.shape == [2, 3]
    assert approx(t.sum(), 21.0)

# ── metadata ──────────────────────────────────────────────────────────────────

def test_rank_size_device():
    t = Tensor.zeros([2, 3, 4])
    assert t.rank == 3
    assert t.size == 24
    assert t.device == "cpu"
    assert not t.is_cuda

def test_repr():
    s = repr(Tensor.zeros([2, 2]))
    assert "shape=[2, 2]" in s
    assert "cpu" in s

# ── tolist / numpy ────────────────────────────────────────────────────────────

def test_tolist():
    t = Tensor.from_list([1, 2, 3], [3])
    assert t.tolist() == [1.0, 2.0, 3.0]

def test_numpy_roundtrip():
    np = pytest.importorskip("numpy")
    data = [1.0, 2.0, 3.0, 4.0]
    t = Tensor.from_list(data, [2, 2])
    arr = t.numpy()
    assert arr.shape == (2, 2)
    assert list(arr.flatten()) == data

# ── arithmetic ────────────────────────────────────────────────────────────────

def test_add_tensor():
    a = Tensor.from_list([1, 2, 3], [3])
    b = Tensor.from_list([4, 5, 6], [3])
    assert (a + b).sum() == 21.0

def test_sub_tensor():
    a = Tensor.from_list([4, 5, 6], [3])
    b = Tensor.from_list([1, 2, 3], [3])
    assert (a - b).sum() == 9.0

def test_mul_tensor():
    a = Tensor.from_list([1, 2, 3], [3])
    b = Tensor.from_list([4, 5, 6], [3])
    assert (a * b).sum() == 32.0

def test_add_scalar():
    a = Tensor.from_list([1, 2, 3], [3])
    assert (a + 10).sum() == 36.0

def test_mul_scalar():
    a = Tensor.from_list([1, 2, 3], [3])
    assert (a * 2).sum() == 12.0

def test_matmul():
    # [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    a = Tensor.from_list([1, 2, 3, 4], [2, 2])
    eye = Tensor.from_list([1, 0, 0, 1], [2, 2])
    c = a @ eye
    assert c.shape == [2, 2]
    assert approx(c.sum(), 10.0)

# ── reductions ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample():
    return Tensor.from_list([3, 1, 4, 1, 5, 9, 2, 6], [8])

def test_sum(sample):    assert approx(sample.sum(), 31.0)
def test_mean(sample):   assert approx(sample.mean(), 31.0 / 8)
def test_min(sample):    assert approx(sample.min(), 1.0)
def test_max(sample):    assert approx(sample.max(), 9.0)

# ── reshape / flatten ─────────────────────────────────────────────────────────

def test_reshape():
    t = Tensor.from_list(list(range(1, 7)), [6])
    r = t.reshape([2, 3])
    assert r.shape == [2, 3]
    assert approx(r.sum(), 21.0)

def test_reshape_wrong_size():
    t = Tensor.from_list([1, 2, 3, 4], [4])
    with pytest.raises(RuntimeError):
        t.reshape([3, 2])

def test_flatten():
    t = Tensor.ones([2, 3, 4])
    f = t.flatten()
    assert f.shape == [24]

def test_squeeze_unsqueeze_roundtrip():
    t = Tensor.from_list([1, 2, 3, 4], [2, 2])
    u = t.unsqueeze(1)
    assert u.shape == [2, 1, 2]
    s = u.squeeze(1)
    assert s.shape == [2, 2]
    assert approx(s.sum(), 10.0)

def test_unsqueeze_invalid_axis():
    t = Tensor.ones([3])
    with pytest.raises(RuntimeError):
        t.unsqueeze(5)

# ── fused ops ─────────────────────────────────────────────────────────────────

def test_scale_shift():
    t = Tensor.from_list([1, 2, 3, 4], [4])
    assert approx(t.scale_shift(2.0, 1.0).sum(), 24.0)

def test_fused_add_mul():
    a = Tensor.from_list([1, 2, 3, 4], [4])
    b = Tensor.ones([4])
    result = a.fused_add_mul(b, 2.0)   # (a+b)*2 = [4,6,8,10]
    assert approx(result.sum(), 28.0)

# ── GPU ───────────────────────────────────────────────────────────────────────

def test_gpu_ones():
    t = Tensor.ones([1024], device="cuda")
    assert t.is_cuda
    assert approx(t.sum(), 1024.0)

def test_gpu_to_cpu():
    g = Tensor.full([16], 3.0, device="cuda")
    c = g.cpu()
    assert not c.is_cuda
    assert approx(c.sum(), 48.0)

def test_cpu_to_gpu():
    c = Tensor.from_list([1, 2, 3, 4], [4])
    g = c.cuda()
    assert g.is_cuda
    assert approx(g.sum(), 10.0)

def test_gpu_add():
    a = Tensor.from_list([1, 2, 3, 4], [4], device="cuda")
    b = Tensor.from_list([4, 3, 2, 1], [4], device="cuda")
    assert approx((a + b).sum(), 20.0)

def test_gpu_reshape():
    t = Tensor.from_list(list(range(1, 7)), [6], device="cuda")
    r = t.reshape([2, 3])
    assert r.shape == [2, 3]
    assert approx(r.sum(), 21.0)

def test_gpu_reduction():
    t = Tensor.from_list([5, 1, 3, 2, 4], [5], device="cuda")
    assert approx(t.min(), 1.0)
    assert approx(t.max(), 5.0)
    assert approx(t.mean(), 3.0)

def test_copy_independence():
    a = Tensor.from_list([1, 2, 3], [3])
    b = a.copy()
    b.fill(0.0)
    assert approx(a.sum(), 6.0)
    assert approx(b.sum(), 0.0)
