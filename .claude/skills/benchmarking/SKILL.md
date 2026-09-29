---
name: benchmarking
description: Run and interpret OpenMat's five benchmark harnesses (bench_vs, bench_rank_sweep, bench_omp, bench_inplace, the HostPool A/B). Use when benchmarking OpenMat, comparing against NumPy/PyTorch, diagnosing a performance regression, or updating benchmark_report.md.
---

# Benchmarking OpenMat

Five harnesses, all requiring a **Release** build (`build-release/`) — Debug numbers are meaningless — and a `bench-env` venv holding NumPy and PyTorch, which the library itself does not need:

- [scripts/bench_vs.py](../../../scripts/bench_vs.py) — the main cross-framework table (CPU + CUDA + transfers). `--quick` for a smoke run, `--no-cuda` to skip the device.
- [scripts/bench_rank_sweep.py](../../../scripts/bench_rank_sweep.py) — the same 16 M buffer reshaped to ranks 1–5, which is what verifies the contiguous fast path is actually engaging. `bench_vs.py` only ever uses rank-1 shapes and would not notice a regression here.
- [scripts/bench_omp.py](../../../scripts/bench_omp.py) — run once per `OMP_NUM_THREADS` value; the 1-vs-20 delta is the OpenMP A/B, and a `1.00×` row is how you confirm an op is *not* parallelized (`min`/`max`, `sum`; `apply`/`apply_binary` are parallel since the views work, 2026-09-29).
- [scripts/bench_inplace.py](../../../scripts/bench_inplace.py) — the in-place / `out=` families against the allocating forms, per op and over a 32-step chain. Every case builds its own operands: an earlier draft that reused one destination across the three ops at each size reported a CUDA `relu_` "regression" of 0.38× that does not exist (1.05× re-measured).
- `OPENMAT_HOST_CACHE_BYTES=0` — restores the pre-`HostPool` allocator, the A/B behind benchmark_report.md §3.

## Two traps that have already produced wrong conclusions in the reports

- **`bench_vs.py`'s `transfer/*` rows at 16 M are not transfer measurements.** They run last, after the process has accumulated `HostPool`, `PinnedHostPool`, PyTorch's CUDA caching allocator and every live operand from the CUDA sweep; on a unified-memory part that pressure lands on the copy. The row reported OpenMat's 64 MB D2H at 39.8 ms while a fresh process measures **1.136 ms / 59.1 GB/s, identical to PyTorch**. The tell is that PyTorch's own D2H degrades alongside it in the same run. Measure transfers in a dedicated process.
- **Re-measure before diagnosing.** A block-size fix once moved GPU `add` from 159 to 220 GB/s without the report being re-run, and the next round of work was aimed at a bottleneck that no longer existed. Every number in [benchmark_report.md](../../../benchmark_report.md) carries the date of the run that produced it for this reason.
