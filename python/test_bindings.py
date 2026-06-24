"""
Smoke-test for the Python bindings.
Run from the repo root: python python/test_bindings.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from openmat import Tensor

def check(name, got, expected, tol=1e-4):
    ok = abs(got - expected) < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got}, expected={expected}")
    if not ok:
        sys.exit(1)

def check_shape(name, t, expected):
    ok = t.shape == list(expected)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: shape={t.shape}, expected={list(expected)}")
    if not ok:
        sys.exit(1)

print("── factories ──────────────────────────────────────────")
z = Tensor.zeros([3, 4])
check_shape("zeros shape", z, [3, 4])
check("zeros sum", z.sum(), 0.0)

o = Tensor.ones([2, 5])
check("ones sum", o.sum(), 10.0)

f = Tensor.full([4], 3.0)
check("full sum", f.sum(), 12.0)

fl = Tensor.from_list([1, 2, 3, 4, 5], [5])
check("from_list sum", fl.sum(), 15.0)

print("── repr ────────────────────────────────────────────────")
print(f"  {fl}")
print(f"  rank={fl.rank}, size={fl.size}, device={fl.device}")

print("── tolist / numpy ──────────────────────────────────────")
lst = fl.tolist()
assert lst == [1.0, 2.0, 3.0, 4.0, 5.0], lst
print(f"  [PASS] tolist: {lst}")

try:
    import numpy as np
    arr = fl.numpy()
    assert list(arr) == [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"  [PASS] numpy: {arr}")
    t_np = Tensor.from_numpy(arr.reshape(1, 5))
    check_shape("from_numpy shape", t_np, [1, 5])
except ImportError:
    print("  [SKIP] numpy not installed")

print("── arithmetic (CPU) ────────────────────────────────────")
a = Tensor.from_list([1, 2, 3], [3])
b = Tensor.from_list([4, 5, 6], [3])
check("add sum", (a + b).sum(), 21.0)
check("sub sum", (b - a).sum(), 9.0)
check("mul sum", (a * b).sum(), 32.0)
check("add scalar", (a + 10).sum(), 36.0)
check("mul scalar", (a * 2).sum(), 12.0)

print("── reductions (CPU) ────────────────────────────────────")
t = Tensor.from_list([3, 1, 4, 1, 5, 9, 2, 6], [8])
check("sum",  t.sum(),  31.0)
check("mean", t.mean(), 31.0 / 8)
check("min",  t.min(),  1.0)
check("max",  t.max(),  9.0)

print("── reshape / flatten ───────────────────────────────────")
r = Tensor.from_list(list(range(1, 7)), [6])
r2 = r.reshape([2, 3])
check_shape("reshape 2x3", r2, [2, 3])
check("reshape sum", r2.sum(), 21.0)

flat = r2.flatten()
check_shape("flatten", flat, [6])

u = flat.unsqueeze(0)
check_shape("unsqueeze", u, [1, 6])
s = u.squeeze(0)
check_shape("squeeze", s, [6])

print("── scale_shift / fused ─────────────────────────────────")
t2 = Tensor.from_list([1, 2, 3, 4], [4])
ss = t2.scale_shift(2.0, 1.0)
check("scale_shift sum", ss.sum(), 24.0)

t3 = Tensor.from_list([1, 1, 1, 1], [4])
fam = t2.fused_add_mul(t3, 2.0)   # (t2+t3)*2 = [4,6,8,10]
check("fused_add_mul sum", fam.sum(), 28.0)

print("── GPU (requires CUDA) ─────────────────────────────────")
try:
    g = Tensor.ones([1024], device="cuda")
    assert g.is_cuda
    check("GPU ones sum", g.sum(), 1024.0)

    gc = g.cpu()
    assert not gc.is_cuda
    check("GPU→CPU sum", gc.sum(), 1024.0)

    a_gpu = Tensor.from_list([1,2,3,4], [4], device="cuda")
    b_gpu = Tensor.from_list([4,3,2,1], [4], device="cuda")
    check("GPU add sum", (a_gpu + b_gpu).sum(), 20.0)

    r_gpu = Tensor.from_list(list(range(1,7)), [6], device="cuda")
    check("GPU reshape sum", r_gpu.reshape([2,3]).sum(), 21.0)
    print("  [PASS] GPU tests passed")
except Exception as e:
    print(f"  [SKIP] GPU: {e}")

print()
print("All tests passed.")
