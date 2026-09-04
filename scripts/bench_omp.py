#!/usr/bin/env python3
"""
CPU OpenMP scaling for the elementwise ops (benchmark_report.md §7).

Run twice, once with OMP_NUM_THREADS=1 and once unset, and compare: the
single-thread column is what the ops were before the `parallel for` landed,
the multi-thread column is what they are now.  NumPy is timed in the same
process as a reference.

    for t in 1 20; do
      OMP_NUM_THREADS=$t OPENMAT_LIB=build-release/OpenMat.so PYTHONPATH=python \
        bench-env/bin/python scripts/bench_omp.py
    done
"""
from __future__ import annotations

import os
import statistics
import time

import numpy as np
import openmat as om

SIZES = [1 << 12, 1 << 16, 1 << 20, 1 << 24]


def measure(fn, *, min_time=0.05, trials=7):
    for _ in range(3):
        fn()

    reps = 1
    while reps < 1_000_000:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        dt = time.perf_counter() - t0
        if dt >= min_time:
            break
        reps = int(reps * max(2.0, min(50.0, min_time / max(dt, 1e-9)))) + 1

    samples = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        samples.append((time.perf_counter() - t0) / reps)
    return min(samples)


def main() -> int:
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"=== OMP_NUM_THREADS={threads} ===")
    print(f"{'op':22s} {'N':>10s} {'om ms':>10s} {'np ms':>10s} {'om/np':>8s}")

    for n in SIZES:
        a_np = (np.arange(n, dtype=np.float32) % 97.0) + 1.0
        b_np = (np.arange(n, dtype=np.float32) % 31.0) + 1.0
        a = om.from_numpy(a_np)
        b = om.from_numpy(b_np)

        cases = {
            "add (tensor)":  (lambda a=a, b=b: a.add(b),
                              lambda x=a_np, y=b_np: x + y),
            "sub (tensor)":  (lambda a=a, b=b: a.sub(b),
                              lambda x=a_np, y=b_np: x - y),
            "mul (tensor)":  (lambda a=a, b=b: a.mul(b),
                              lambda x=a_np, y=b_np: x * y),
            "div (tensor)":  (lambda a=a, b=b: a.div(b),
                              lambda x=a_np, y=b_np: x / y),
            "add (scalar)":  (lambda a=a: a.add(1.5),
                              lambda x=a_np: x + 1.5),
            "mul (scalar)":  (lambda a=a: a.mul(1.5),
                              lambda x=a_np: x * 1.5),
            "min":           (lambda a=a: a.min(), lambda x=a_np: float(x.min())),
            "max":           (lambda a=a: a.max(), lambda x=a_np: float(x.max())),
            "sum":           (lambda a=a: a.sum(), lambda x=a_np: float(x.sum())),
            "relu":          (lambda a=a: a.relu(),
                              lambda x=a_np: np.maximum(x, 0.0)),
            "x*s+t":         (lambda a=a: a.scale_shift(2.0, 1.0),
                              lambda x=a_np: x * 2.0 + 1.0),
        }
        for name, (om_fn, np_fn) in cases.items():
            t_om = measure(om_fn)
            t_np = measure(np_fn)
            print(f"{name:22s} {n:>10d} {t_om * 1e3:10.4f} {t_np * 1e3:10.4f} "
                  f"{t_om / t_np:8.2f}")
        print()
        del a, b
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
