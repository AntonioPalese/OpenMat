#!/usr/bin/env python3
"""
OpenMat vs NumPy vs PyTorch — micro-benchmark harness.

Runs the same set of elementwise, fused, reduction and linear-algebra
operations through the three libraries, on CPU and (where supported) CUDA,
and emits a JSON result file plus a human-readable table.

Usage:
    OPENMAT_LIB=build-release/OpenMat.so \
    PYTHONPATH=python python scripts/bench_vs.py --out bench_results.json

Timing methodology: each case is warmed up, then measured as `reps` batched
runs repeated `trials` times; the reported number is the *minimum* per-op time
across trials (least noise-contaminated estimate).  CUDA cases synchronize the
device once per batch, outside the timed loop's inner repetition, so launch
overhead is amortised the same way for every library.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from typing import Callable, Optional

import numpy as np
import openmat as om
import torch

# --------------------------------------------------------------------------
# timing core
# --------------------------------------------------------------------------


def _time_batch(fn: Callable[[], None], sync: Optional[Callable[[], None]],
                reps: int) -> float:
    """Seconds for `reps` calls of fn, including one trailing sync."""
    if sync:
        sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if sync:
        sync()
    return time.perf_counter() - t0


def measure(fn: Callable[[], None], sync: Optional[Callable[[], None]] = None,
            *, min_time: float = 0.05, trials: int = 7,
            max_reps: int = 100_000) -> dict:
    """Return per-call seconds: min / median across `trials` batches."""
    # warm-up (allocator pools, JIT, first-touch page faults)
    for _ in range(3):
        fn()
    if sync:
        sync()

    # calibrate reps so a batch lasts at least min_time
    reps = 1
    while reps < max_reps:
        dt = _time_batch(fn, sync, reps)
        if dt >= min_time:
            break
        # extrapolate, with a safety factor and a cap on the growth per step
        grow = max(2.0, min(50.0, min_time / max(dt, 1e-9)))
        reps = min(max_reps, int(reps * grow) + 1)

    samples = [_time_batch(fn, sync, reps) / reps for _ in range(trials)]
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "reps": reps,
        "trials": trials,
    }


# --------------------------------------------------------------------------
# case registry
# --------------------------------------------------------------------------

CASES: list[dict] = []


def case(group: str, name: str, shape, *, bytes_moved=None, flops=None):
    """Register a benchmark case; the decorated fn builds the per-library closures."""
    def deco(builder):
        CASES.append({
            "group": group, "name": name, "shape": list(shape),
            "bytes": bytes_moved, "flops": flops, "builder": builder,
        })
        return builder
    return deco


# --------------------------------------------------------------------------
# operand construction
# --------------------------------------------------------------------------


def operands(shape, device: str):
    """Matching (a, b) pairs for every library, from the same source data."""
    n = int(np.prod(shape))
    a_np = np.arange(n, dtype=np.float32).reshape(shape) % 97.0 + 1.0
    b_np = (np.arange(n, dtype=np.float32).reshape(shape) % 31.0) + 1.0

    tdev = "cuda" if device == "cuda" else "cpu"
    out = {
        "np": (a_np, b_np),
        "torch": (torch.from_numpy(a_np).to(tdev), torch.from_numpy(b_np).to(tdev)),
        "om": (om.from_numpy(a_np).to(device), om.from_numpy(b_np).to(device)),
    }
    if device == "cuda":
        torch.cuda.synchronize()
        om.synchronize()
        out.pop("np")  # NumPy has no GPU backend
    return out


def syncs(device: str):
    if device != "cuda":
        return {"np": None, "torch": None, "om": None}
    return {"torch": torch.cuda.synchronize, "om": om.synchronize}


# --------------------------------------------------------------------------
# the cases
# --------------------------------------------------------------------------

ELEMENTWISE_SHAPES = [(1 << 12,), (1 << 16,), (1 << 20,), (1 << 24,)]
MATMUL_SHAPES = [(128, 128), (512, 512), (1024, 1024)]
TRANSPOSE_SHAPES = [(512, 512), (2048, 2048)]


def _register():
    for shape in ELEMENTWISE_SHAPES:
        n = int(np.prod(shape))

        @case("elementwise", "add", shape, bytes_moved=3 * 4 * n, flops=n)
        def _(ops, s=None):
            return {
                "np": (lambda a=ops["np"][0], b=ops["np"][1]: a + b) if "np" in ops else None,
                "torch": lambda a=ops["torch"][0], b=ops["torch"][1]: a + b,
                "om": lambda a=ops["om"][0], b=ops["om"][1]: a.add(b),
            }

        @case("elementwise", "mul", shape, bytes_moved=3 * 4 * n, flops=n)
        def _(ops):
            return {
                "np": (lambda a=ops["np"][0], b=ops["np"][1]: a * b) if "np" in ops else None,
                "torch": lambda a=ops["torch"][0], b=ops["torch"][1]: a * b,
                "om": lambda a=ops["om"][0], b=ops["om"][1]: a.mul(b),
            }

        @case("fused", "(a+b)*s", shape, bytes_moved=3 * 4 * n, flops=2 * n)
        def _(ops):
            return {
                "np": (lambda a=ops["np"][0], b=ops["np"][1]: (a + b) * 2.5) if "np" in ops else None,
                "torch": lambda a=ops["torch"][0], b=ops["torch"][1]: (a + b) * 2.5,
                "om": lambda a=ops["om"][0], b=ops["om"][1]: a.fused_add_mul(b, 2.5),
            }

        @case("fused", "x*s+t", shape, bytes_moved=2 * 4 * n, flops=2 * n)
        def _(ops):
            return {
                "np": (lambda a=ops["np"][0]: a * 2.0 + 1.0) if "np" in ops else None,
                "torch": lambda a=ops["torch"][0]: a * 2.0 + 1.0,
                "om": lambda a=ops["om"][0]: a.scale_shift(2.0, 1.0),
            }

        @case("unary", "relu", shape, bytes_moved=2 * 4 * n, flops=n)
        def _(ops):
            return {
                "np": (lambda a=ops["np"][0]: np.maximum(a, 0.0)) if "np" in ops else None,
                "torch": lambda a=ops["torch"][0]: torch.relu(a),
                "om": lambda a=ops["om"][0]: a.relu(),
            }

        @case("reduction", "sum", shape, bytes_moved=4 * n, flops=n)
        def _(ops):
            return {
                "np": (lambda a=ops["np"][0]: float(a.sum())) if "np" in ops else None,
                "torch": lambda a=ops["torch"][0]: float(a.sum()),
                "om": lambda a=ops["om"][0]: a.sum(),
            }

    for shape in MATMUL_SHAPES:
        m = shape[0]

        @case("linalg", "matmul", shape, flops=2 * m ** 3)
        def _(ops):
            return {
                "np": (lambda a=ops["np"][0], b=ops["np"][1]: a @ b) if "np" in ops else None,
                "torch": lambda a=ops["torch"][0], b=ops["torch"][1]: a @ b,
                "om": lambda a=ops["om"][0], b=ops["om"][1]: a.matmul(b),
            }

    for shape in TRANSPOSE_SHAPES:
        n = int(np.prod(shape))

        # NumPy/Torch .T is a view; force the materialisation OpenMat performs.
        @case("shape", "transpose", shape, bytes_moved=2 * 4 * n)
        def _(ops):
            return {
                "np": (lambda a=ops["np"][0]: np.ascontiguousarray(a.T)) if "np" in ops else None,
                "torch": lambda a=ops["torch"][0]: a.t().contiguous(),
                "om": lambda a=ops["om"][0]: a.transpose(),
            }


_register()


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(device: str, quick: bool) -> list[dict]:
    sync = syncs(device)
    results = []
    kw = dict(min_time=0.02, trials=3) if quick else dict(min_time=0.05, trials=7)

    # group cases by shape so operands are built once per shape
    by_shape: dict[tuple, list[dict]] = {}
    for c in CASES:
        by_shape.setdefault(tuple(c["shape"]), []).append(c)

    for shape, cases in by_shape.items():
        try:
            ops = operands(shape, device)
        except Exception as exc:                                # pragma: no cover
            print(f"  !! operands {shape} on {device}: {exc}", file=sys.stderr)
            continue

        for c in cases:
            fns = c["builder"](ops)
            row = {"group": c["group"], "name": c["name"], "shape": list(shape),
                   "device": device, "bytes": c["bytes"], "flops": c["flops"],
                   "libs": {}}
            label = f"{c['group']}/{c['name']} {tuple(shape)} [{device}]"
            for lib, fn in fns.items():
                if fn is None:
                    continue
                try:
                    row["libs"][lib] = measure(fn, sync.get(lib), **kw)
                except Exception as exc:
                    row["libs"][lib] = {"error": f"{type(exc).__name__}: {exc}"}
            results.append(row)
            best = {k: v.get("min") for k, v in row["libs"].items()}
            print(f"  {label:44s} " + "  ".join(
                f"{k}={v * 1e6:9.2f}us" if v else f"{k}=ERR" for k, v in best.items()))
        del ops

    return results


def transfer_bench(quick: bool) -> list[dict]:
    """Host<->device copy bandwidth, OpenMat vs PyTorch."""
    out = []
    kw = dict(min_time=0.02, trials=3) if quick else dict(min_time=0.05, trials=5)
    for n in (1 << 16, 1 << 20, 1 << 24):
        a_np = np.zeros(n, dtype=np.float32)
        t_h = torch.from_numpy(a_np)
        o_h = om.from_numpy(a_np)
        t_d = t_h.cuda()
        o_d = o_h.cuda()
        torch.cuda.synchronize(); om.synchronize()

        for direction, fns in (
            ("h2d", {"torch": lambda: t_h.to("cuda"), "om": lambda: o_h.cuda()}),
            ("d2h", {"torch": lambda: t_d.cpu(),      "om": lambda: o_d.cpu()}),
        ):
            row = {"group": "transfer", "name": direction, "shape": [n],
                   "device": "cuda", "bytes": 4 * n, "flops": None, "libs": {}}
            for lib, fn in fns.items():
                sync = torch.cuda.synchronize if lib == "torch" else om.synchronize
                row["libs"][lib] = measure(fn, sync, **kw)
            out.append(row)
            print(f"  transfer/{direction} ({n},) [cuda]".ljust(46) + "  ".join(
                f"{k}={v['min'] * 1e6:9.2f}us" for k, v in row["libs"].items()))
    return out


def env_info() -> dict:
    info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "openmat": om.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
        info["cuda_runtime"] = torch.version.cuda
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--quick", action="store_true", help="fewer trials, shorter batches")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()

    env = env_info()
    print("=== environment ===")
    for k, v in env.items():
        print(f"  {k:18s} {v}")

    rows = []
    print("\n=== CPU ===")
    rows += run("cpu", args.quick)

    if not args.no_cuda and torch.cuda.is_available() and om.cuda_is_available():
        print("\n=== CUDA ===")
        rows += run("cuda", args.quick)
        print("\n=== transfers ===")
        rows += transfer_bench(args.quick)

    with open(args.out, "w") as fh:
        json.dump({"env": env, "results": rows}, fh, indent=2)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
