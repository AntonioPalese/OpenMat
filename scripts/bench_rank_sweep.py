#!/usr/bin/env python3
"""
Rank sweep for the contiguous fast path (benchmark_report.md §8).

The same 16 M-element buffer, reshaped so the launcher would pick a different
rank-specialized layout, timed end-to-end through the Python bindings
(allocation and synchronization included) with the same min-of-N-batches
method as scripts/bench_vs.py.

    OPENMAT_LIB=build-release/OpenMat.so PYTHONPATH=python \
      bench-env/bin/python scripts/bench_rank_sweep.py
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import openmat as om

N = 1 << 24
SHAPES = [
    (N,),
    (4096, 4096),
    (256, 256, 256),
    (64, 64, 64, 64),
    (16, 16, 16, 16, 256),
]


def measure(fn, sync, *, min_time=0.05, trials=7):
    for _ in range(3):
        fn()
    sync()

    reps = 1
    while reps < 100_000:
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        dt = time.perf_counter() - t0
        if dt >= min_time:
            break
        reps = int(reps * max(2.0, min(50.0, min_time / max(dt, 1e-9)))) + 1

    samples = []
    for _ in range(trials):
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        samples.append((time.perf_counter() - t0) / reps)
    return {"min": min(samples), "median": statistics.median(samples), "reps": reps}


def main() -> int:
    if not om.cuda_is_available():
        print("no CUDA device")
        return 1

    a_np = (np.arange(N, dtype=np.float32) % 97.0) + 1.0
    b_np = (np.arange(N, dtype=np.float32) % 31.0) + 1.0

    print(f"{'op':18s} {'shape':24s} {'rank':>4s} {'min us':>10s} {'GB/s':>8s}")
    for shape in SHAPES:
        # reshape is a deep copy in this library: do it once, outside the loop
        a = om.from_numpy(a_np).cuda().reshape(*shape)
        b = om.from_numpy(b_np).cuda().reshape(*shape)
        om.synchronize()

        cases = {
            "add": (lambda a=a, b=b: a.add(b), 3),
            "relu": (lambda a=a: a.relu(), 2),
            "fused (a+b)*s": (lambda a=a, b=b: a.fused_add_mul(b, 2.5), 3),
        }
        for name, (fn, traffic) in cases.items():
            r = measure(fn, om.synchronize)
            gbs = traffic * 4 * N / r["min"] / 1e9
            print(f"{name:18s} {str(shape):24s} {len(shape):>4d} "
                  f"{r['min'] * 1e6:10.1f} {gbs:8.1f}")
        del a, b
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
