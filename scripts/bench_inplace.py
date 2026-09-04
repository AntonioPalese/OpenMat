#!/usr/bin/env python3
"""
In-place / destination-provided ops vs the allocating forms.

Answers two separate questions that the same table is easy to conflate:

  * what one op costs — `a + b` (allocate a result, compute, free it) against
    `dst.add_(b)` and `a.add_out(b, dst)` on a buffer that already exists;
  * what a *loop* costs — N iterations of the same op, which is where the
    allocating form pays its cost N times and holds every intermediate alive.

Each case gets its own freshly built operands: reusing one destination across
cases lets an earlier case leave values (or an allocator state) behind that
changes the next one's timing, which is exactly how an earlier draft of this
script produced a 0.38x "regression" that does not exist.

    OPENMAT_LIB=build-release/OpenMat.so PYTHONPATH=python \
      bench-env/bin/python scripts/bench_inplace.py
"""
from __future__ import annotations

import argparse
import statistics
import time

import openmat as om

SIZES = [1 << 10, 1 << 16, 1 << 20, 1 << 24]


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
    return min(samples)


def ops(device):
    """(label, out-of-place, in-place, destination-provided) built per case."""
    def build(n):
        a = om.Tensor.full([n], 1.5, device=device)
        b = om.Tensor.full([n], 2.0, device=device)
        d = om.Tensor.zeros([n], device=device)
        return a, b, d

    return [
        ("add  (tensor)", build,
         lambda a, b, d: a + b,
         lambda a, b, d: d.add_(b),
         lambda a, b, d: a.add_out(b, d)),
        ("mul  (scalar)", build,
         lambda a, b, d: a * 2.0,
         lambda a, b, d: d.mul_(2.0),
         lambda a, b, d: a.mul_out(2.0, d)),
        ("relu", build,
         lambda a, b, d: a.relu(),
         lambda a, b, d: d.relu_(),
         lambda a, b, d: a.relu_out(d)),
    ]


def single_op_table(device, sync):
    print(f"\n── {device}: one op ──")
    print(f"{'n':>10}  {'op':14} {'alloc us':>10} {'in-place us':>12} "
          f"{'out= us':>10} {'in-place':>9} {'out=':>7}")
    for n in SIZES:
        for label, build, oop, inplace, outp in ops(device):
            a, b, d = build(n)
            t_alloc = measure(lambda: oop(a, b, d), sync)
            t_in = measure(lambda: inplace(a, b, d), sync)
            t_out = measure(lambda: outp(a, b, d), sync)
            print(f"{n:>10}  {label:14} {t_alloc*1e6:10.2f} {t_in*1e6:12.2f} "
                  f"{t_out*1e6:10.2f} {t_alloc/t_in:8.2f}x {t_alloc/t_out:6.2f}x")


def loop_table(device, sync, steps=32):
    """A `steps`-long chain, the shape a training loop actually has."""
    print(f"\n── {device}: a {steps}-step chain (w += g each step) ──")
    print(f"{'n':>10}  {'alloc us':>10} {'in-place us':>12} {'speedup':>8}")
    for n in SIZES:
        g = om.Tensor.full([n], 0.5, device=device)

        def allocating():
            w = om.Tensor.zeros([n], device=device)
            for _ in range(steps):
                w = w + g
            return w

        w_ip = om.Tensor.zeros([n], device=device)

        def in_place():
            w_ip.fill_(0.0)
            for _ in range(steps):
                w_ip.add_(g)
            return w_ip

        t_a = measure(allocating, sync, trials=5)
        t_i = measure(in_place, sync, trials=5)
        print(f"{n:>10}  {t_a*1e6:10.2f} {t_i*1e6:12.2f} {t_a/t_i:7.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    devices = ["cpu"]
    if not args.no_cuda and om.cuda_is_available():
        devices.append("cuda")

    for device in devices:
        sync = om.synchronize if device == "cuda" else (lambda: None)
        single_op_table(device, sync)
        loop_table(device, sync)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
