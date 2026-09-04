<p align="center">
  <img src="images/logo.png" alt="OpenMat Logo" width="300"/>
</p>

# OpenMat

**High-performance CUDA tensor framework in C++/CUDA** — rank-specialized kernels, RAII GPU memory management, stream-ordered allocation, and N-dimensional tensor operations, with a ctypes Python package on top.

> Not a wrapper around cuBLAS/CUDNN. Every kernel is written from scratch.

![Language](https://img.shields.io/badge/language-C%2B%2B17%20%2F%20CUDA-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-ctypes%20bindings-yellow)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## What it is

OpenMat is a CUDA tensor library built to explore the design space of GPU kernel authoring, memory allocator architecture, and Python FFI — from first principles, without relying on cuBLAS or other vendor-provided primitives.

The goal is not to beat PyTorch. The goal is to understand exactly what PyTorch is doing under the hood, and to make deliberate choices at each layer.

---

## Performance

Timing lives in three GoogleTest suites plus one cross-framework script:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..   # Debug numbers are meaningless
make -j$(nproc)
./tests/test_benchmarks     # matmul (fp32/fp16), elementwise, reductions
./tests/test_stream_perf    # the stream experiments tabulated below
./tests/test_stress         # soak / leak checking

# vs NumPy and PyTorch — needs both installed, see benchmark_report.md
OPENMAT_LIB=build/OpenMat.so PYTHONPATH=python python scripts/bench_vs.py
```

### vs NumPy and PyTorch

Measured on **NVIDIA GB10 (DGX Spark, `sm_121`) · CUDA 13.0 · 20× ARMv9**, Release build,
`float32`, against **NumPy 2.5.2** and **PyTorch 2.14.0+cu130**. Minimum per-op time over
seven calibrated batches; every op out-of-place, as all three libraries are by default.
Full tables, root-cause analysis and reproduction steps in
[benchmark_report.md](benchmark_report.md).

| | OpenMat vs the faster reference |
|---|---|
| GPU elementwise `add`, 16 M elem | **1.02× faster than PyTorch** — 236 GB/s vs 232, at **any rank** |
| GPU fused `(a+b)*s`, 16 M elem | **1.68× faster than PyTorch** — 851 µs vs 1428 µs |
| GPU fused `x*s+t`, 16 M elem | **1.97× faster than PyTorch** — 573 µs vs 1130 µs |
| GPU `relu`, 16 M elem | parity — 573.2 µs vs 569.5 µs |
| GPU `transpose`, 2048² | **1.15× faster than PyTorch** — 205 µs vs 236 µs |
| CPU elementwise `add`, 16 M elem | **2.83× faster than NumPy**, parity with PyTorch |
| CPU fused `x*s+t`, 16 M elem | **3.06× faster than NumPy, 1.07× faster than PyTorch** |
| CPU `sum`, 16 M elem | **1.06× faster than NumPy** — 36.9 GB/s vs 34.8 |
| H2D / D2H transfer | parity, within 5 % |
| GPU `sum`, 16 M elem | 1.16× slower — 317 µs vs 273 µs |
| CPU `min`/`max`, 16 M elem | 2.7× slower than NumPy |
| Per-op dispatch floor | 1.9 µs on CPU, 7.9 µs on CUDA (PyTorch: 0.8 / 3.1 µs) |
| CPU `matmul`, 1024³ | 6.3× slower — 123 GFLOP/s vs 768 GFLOP/s |
| GPU `matmul`, 1024³ | 10.3× slower — 1.64 TFLOP/s vs cuBLAS' 16.9 TFLOP/s |

**The fusion bet pays off.** At 16 M elements PyTorch's `(a + b) * 2.5` materialises a
64 MB intermediate and reads it back; `fused_add_mul` writes one buffer once. `scale_shift`
shows it more sharply still — 1.97×, because `x * 2.0 + 1.0` costs PyTorch two full passes.
The same argument runs in reverse on the CPU, where NumPy's `a * 2.0 + 1.0` is two passes
over 64 MB and `scale_shift` beats both references *while still single-threaded*.

**Elementwise kernels ignored contiguity — every rank now runs at rank-1 speed.**
The rank-specialized launchers map tensor axes onto grid axes, and only at rank 1 does
that leave a warp's 32 lanes contiguous in memory. A rank-2 `dim3(16,16)` block gives each
warp two disjoint runs of 16 elements, a rank-3 `dim3(8,8,8)` block four runs of 8: 64- and
32-byte requests against a 128-byte line. The shape was never load-bearing — every tensor
OpenMat produces is contiguous row-major, since `reshape` and friends deep-copy — so the
launchers now index the buffer linearly and give every rank the rank-1 layout. Kernel time
only, `add` over 16 M elements:

| dtype | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
|---|---|---|---|---|---|
| `float` before | 229.7 | 206.3 | 141.3 | 104.4 | 26.1 |
| `float` after | 229.5 | **228.6** | **232.8** | **229.0** | **231.0** |
| `float16_t` before | 241.8 | 107.5 | 68.0 | | |
| `float16_t` after | 231.5 | **232.1** | **232.2** | | |
| `char` before | 190.8 | 52.7 | 34.0 | | |
| `char` after | **270.1** | **270.2** | **269.8** | | |

GB/s. (`char` reads high throughout because 16 M elements is only 48 MB of traffic against
a 24 MB L2 — compare it with the `char` row above it, not with `float`.) End-to-end through
the Python bindings, allocation and synchronization included, the same 16 M `float32` buffer
now takes **861–865 µs for `add` at every rank from 1 to 5** — a spread of under 1 %, against
7644 µs at rank 5 and 1442 µs at rank 3 before. `TensorView::is_contiguous()` gates the path,
so a strided view (when views land) falls back to the existing kernels instead of reading
the wrong elements.

That is what moved the headline row: GPU `add` at 16 M went from 1.47× *slower* than PyTorch
to marginally faster, `relu` from 1.9× slower to parity, and `x*s+t` from parity to 1.97×
faster — without touching those ops' arithmetic at all.

`char` was the one dtype that needed more than flattening: a lane loading one `char` puts
32 bytes in front of a warp, worth 190.8 GB/s even at rank 1, where the layout was already
ideal. So the fast path packs `4 / sizeof(T)` elements per thread. **Four bytes, not
sixteen** — `float4` vector loads are the standard advice and they measured *worse* here:

| bytes per thread | `float` | `float16_t` | `char` |
|---|---|---|---|
| 1 | | | 193 |
| 2 | | 235 | |
| 4 | **235** | **239** | **236** |
| 16 (`float4`) | 216 | 223 | 223 |

One 16-byte load per thread costs the launch the thread-level parallelism it needs to keep
the memory pipeline full; a grid-stride loop capped at a few waves per SM costs a further
8–12 %. Neither is used. Full analysis in
[benchmark_report.md §8](benchmark_report.md#8-elementwise-kernels-ignored-contiguity--every-rank-now-runs-at-rank-1-speed).

**The CPU gap above 128 KB was the allocator, not the loop — now fixed.** `add` at 16 M
measured 25.2 ms against NumPy's 6.1 ms, because `CpuAllocator` called `malloc` directly
and past glibc's mmap threshold every out-of-place op faulted in all 16 384 pages of its
own result. `CpuAllocator` now recycles host blocks through a size-classed free list
(`om::detail::HostPool`), as NumPy and PyTorch do. The A/B still reproduces:
`OPENMAT_HOST_CACHE_BYTES=0` restores the old behaviour and costs **6.2×** on a 16 M `add`
(1.99 ms → 12.27 ms). It also closed the repeated-transfer number both reports flagged as
the outstanding bottleneck — 100 × 64 MB H2D+D2H round-trips went from **3109 ms (4.3 GB/s)
to 263 ms (51.1 GB/s)**, 11.8×.

**The CPU backend was single-threaded scalar — `add`/`sub`/`mul`/`div` no longer are.**
`DEFINE_BINARY_OPS_CPU`/`DEFINE_UNARY_OPS_CPU` (the CPU side of those four ops, both
tensor⊕tensor and tensor⊕scalar) now wrap their loop in a size-gated
`#pragma omp parallel for schedule(static) if(_total > 65536)` — one change to each of
two macros, and every op they generate benefits together. Below the threshold nothing
changes (measured within 5 % of the single-thread time — no fork/join tax on small
tensors); above it, **1.7–11.6×** faster on a 20-thread reference machine, enough to beat
NumPy outright by 2.9–3.1× at 16 M rather than merely match it. `matmul_cpu` had already
been parallelized the same way. The attempt to do the analogous thing for `min`/`max` —
`#pragma omp simd reduction(min:)/(max:)` — was tried and **reverted**: isolated
measurement showed it 1.6× *slower*, because GCC 13 already auto-vectorizes that loop
shape under `-O3` alone and the explicit reduction clause forces a worse lowering on top
of it. Full numbers in [benchmark_report.md §7](benchmark_report.md#7-cpu-elementwise-ops-were-single-threaded--openmp-closes-most-of-it).
`apply`/`apply_binary` (`relu`, `sigmoid`, `scale_shift`, the `fused_*` family) did not
get this pass and are still single-threaded on CPU regardless of size — they beat NumPy
anyway, on fusion alone, which is why this is a missed opportunity rather than a defect.

**`sum` was a serial accumulate — now fixed too.** The loop-carried FP dependency in
`reduce_sum_cpu` blocked auto-vectorisation, so it ran at one add per FP *latency*: 7.7 GB/s
against NumPy's pairwise SIMD reduction, 4× behind. Splitting the accumulation across **8
independent lanes** merged pairwise at the end breaks the chain — a source-level
restructuring, not a pragma — and takes it to **36.9 GB/s, 1.06× faster than NumPy**. (It is
also slightly more accurate, since the partials merged at the end are of similar magnitude.)
PyTorch still wins by 2.8× there by threading the reduction across 20 cores, which is the
obvious next step.

**`matmul` on CPU closed 68×, and is no longer the structural gap.** `matmul_cpu` was the
textbook `ijk` triple loop indexing `rhs(k, j)` down a column — 1.81 GFLOP/s, 421× off
NumPy. Reordering to `ikj` with 128-wide L2 tiling and an `omp parallel for` over the
independent output rows takes it to **123 GFLOP/s**, 6.3× off OpenBLAS at 1024³ and
*slightly ahead* of NumPy at 128³. What remains is the gap between a tiled C loop and a
hand-tuned microkernel with register blocking, NEON intrinsics and packed panels — a
different implementation class, but a 6× one rather than three orders of magnitude.

**The remaining gaps are the GPU `matmul` and CPU `min`/`max`.** The GPU kernel is tiled but
computes one output element per thread (two shared-memory loads per FMA), does not double
buffer, and does not touch the tensor cores — `test_benchmarks` prices that last one
directly, with fp16 buying only ~1.1× over fp32. `min`/`max` remain a plain scalar loop,
2.7× behind NumPy; a `parallel for` with a cross-thread `reduction(min:)` clause is the
untried lever there (the `simd` clause, a different change, was measured and reverted).

Caveat on the CPU column: PyTorch runs 20 intraop threads. OpenMat's CPU backend is
threaded via OpenMP for `add`/`sub`/`mul`/`div` and `matmul` but still single-threaded for
`min`, `max`, `sum` and the whole fused-op family — so PyTorch-vs-OpenMat on CPU measures
*what a user gets* on those ops, not comparable algorithms. NumPy is the like-for-like
reference there.

---

## CUDA Streams

OpenMat exposes a full stream API on `Tensor<T>`. Every operation has both a synchronous and a stream-aware variant:

```cpp
om::Stream s;

// synchronous (no stream argument — uses null stream internally)
auto c = a + b;

// asynchronous — kernel enqueued on s, host returns immediately
auto c = a.add(b, s);
s.synchronize();  // block host until work on s is done
```

All methods share the same kernel dispatch code. The no-stream variants delegate to the stream version with `Stream::default_stream()` (a non-owning null-stream wrapper), so there is no code duplication.

### When streams help — and when they do not

Benchmarked with `tests/test_stream_perf.cpp` on two machines: **NVIDIA GeForce RTX 4060 · CUDA 11.5**
(the original numbers) and **NVIDIA GB10 (DGX Spark, `sm_121`) · CUDA 13.0**, both in Release.
The GB10 column is the median of 9 consecutive runs, re-measured after the contiguous
fast path landed — which roughly halved the kernel times every one of these tests is
built on, so the absolute numbers moved even where the conclusion did not.

**The results do not transfer between the two.** Two of the five conclusions invert — the
sequential-chain win disappears and the parallel fan-out, a wash on the 4060, becomes the
largest gain. Full analysis in [stream_perf_report.md](stream_perf_report.md).

#### Single op, sync after each iteration — no improvement on either

| Variant | RTX 4060 | GB10 |
|---|---|---|
| `operator+` (sync) | 0.42 ms · 1.00× | 0.22 ms · 1.00× |
| `add(default_stream())` | 0.42 ms · 1.00× | 0.22 ms · ~1.01× |
| `add(Stream s)` + sync | 0.40 ms · ~1.05× | 0.21 ms · ~1.03× |

When the host blocks after every single operation the round-trip cost is identical regardless of whether a stream is used. The stream wrapper adds no measurable overhead, but it also cannot help if you synchronize immediately.

#### Sequential chain of dependent ops — **2.68× on the 4060, ~1.1× on GB10**

| Variant | RTX 4060 (100 adds, 8 MB) | GB10 |
|---|---|---|
| Sync after every op | 45.38 ms · 1.00× | 5.75 ms · 1.00× |
| One stream, one sync at the end | 16.94 ms · **2.68×** | 5.29 ms · 1.09× |

On the 4060 this is the highest-impact use case. With 100 explicit synchronizations the host stalls 100 times; each stall costs ~0.28 ms of scheduling overhead. Enqueuing all 100 kernels on a single stream and syncing once eliminates 99 of those stalls. The GPU's internal ordering guarantees correctness even with data dependencies between ops.

On GB10 the same 100 syncs cost well under a millisecond in total, so the chain is bounded by kernel work rather than by host stalls and batching buys ~9%. The principle — streams help once you stop synchronizing — still holds; its magnitude is entirely a function of what one sync costs on the platform, and on a unified-memory part that is close to nothing. Note the chain itself got 2.5× faster in absolute terms (14.6 → 5.75 ms) purely from the contiguous fast path, which is why the ratio moved at all.

The pattern in practice:

```cpp
om::Stream s;
Tensor<float> x = input;
for (int i = 0; i < 100; ++i)
    x = x.add(bias, s);   // all 100 kernels enqueued without blocking
s.synchronize();           // one round-trip at the end
```

#### Parallel fan-out of independent ops — a wash on the 4060, the biggest win on GB10

| K | RTX 4060 (seq → K streams) | GB10 (seq → K streams) |
|---|---|---|
| 2 | 0.15 → 0.14 ms · ~1.05× | 0.14 → 0.04 ms · **3.29×** |
| 4 | 0.30 → 0.31 ms · ~0.97× | 0.10 → 0.09 ms · ~1.04× |
| 8 | 0.73 → 0.69 ms · ~1.06× | 1.47 → 0.28 ms · **5.19×** |
| 16 | 1.46 → 1.38 ms · ~1.06× | 1.60 → 1.73 ms · ~0.93× |

On the 4060, launching K independent kernels on K streams does not produce meaningful speedup when each kernel already saturates the available memory bandwidth: that GPU has one compute pipeline, and `mul` on 4 MB is entirely memory-bound.

That reasoning does not describe GB10, where eight independent muls on eight streams run 5.2× faster than sequentially. The shape is emphatically non-monotonic — K=2 gives 3.3×, K=4 falls back to ~1.04×, K=8 peaks at 5.2×, K=16 regresses to ~0.93× — and the test times a single un-warmed run per K, so some of that irregularity is measurement rather than hardware. Over nine runs, though, the spread per row is narrow enough (K=8 spans 4.9–5.7×, K=16 spans 0.89–1.10×) that the shape itself is real: K=16 sits below 1.0× in eight of nine runs, which was not true before and is worth a closer look than a suite timing one un-warmed run per K can give it.

#### Compute + transfer overlap — **1.12× on the 4060, ~4× on GB10**

| Variant | RTX 4060 (20 rounds, 16 MB) | GB10 |
|---|---|---|
| Serialized (H2D → sync → compute → sync) | 37.21 ms · 1.00× | 40.3 ms · 1.00× |
| Overlapped (stream_copy ∥ stream_compute) | 33.11 ms · **1.12×** | 8.77 ms · **~4.6×** |

The RTX 4060 has a dedicated DMA copy engine that operates independently from the compute SMs. Assigning H2D transfers to one stream and compute work to another lets both run simultaneously:

```
Serialized:  [H2D -------][compute -------][H2D -------][compute -------]
Overlapped:  [H2D -------]
                  [compute -------]
                               [H2D -------]
                                    [compute -------]
```

The theoretical maximum speedup is ~2× (total time drops to `max(H2D, compute)` instead of `H2D + compute`). The observed 12% is lower because H2D and compute are similar in duration on this workload, leaving limited asymmetry to exploit. In inference pipelines that alternate between data loading and computation the gain is more pronounced.

GB10's ~4.6× exceeds that ceiling, which means the ratio is still measuring a mediocre baseline rather than exceptional overlap: the serialized path is *slower* there than on the 4060 (40.3 ms vs 37.21 ms) because the H2D leg dominates. The cause was never a staging copy, and that is now settled from both directions.

First, pinning was tried and made no difference: `Tensor::to()` pins the destination of a device-to-host transfer (`cudaHostAlloc` via `PinnedCpuAllocator`, [benchmark_report.md §3](benchmark_report.md#3-the-cpu-gap-above-128-kb-was-the-allocator-not-the-loop--fixed)), which is exactly what would skip a pinned bounce-buffer stage if one were in the way — and it moves nothing on GB10 (58–59 GB/s either way, isolating the copy itself), because NVLink-C2C's unified host/device memory has no such stage to skip in the first place.

Second, the allocator was the whole of it. `test_stress` used to put 100 × 64 MB H2D+D2H round-trips at **4.3 GB/s**, because that test allocates its destination every round and paid the page faults described under [vs NumPy and PyTorch](#vs-numpy-and-pytorch). With `HostPool` recycling those blocks the same test now runs at **51.1 GB/s** — 3109 ms down to 263 ms, 11.8× — and a single isolated 64 MB copy measures 59.1 GB/s, identical to PyTorch. The lever really was the allocator, not pinning. What is left in the serialized baseline above is genuine H2D time, so this row's speedup should now be read as real overlap rather than as a bad baseline.

#### Stream creation overhead — negligible

| Variant | RTX 4060 (256 KB mul, 1000 iters) | GB10 |
|---|---|---|
| Reuse one stream | 0.01 ms | 0.01 ms |
| New stream per call | 0.01 ms | 0.01 ms |

At this scale `cudaStreamCreate` overhead is within measurement noise. For latency-sensitive tight loops (kernel time < 10 µs) the recommendation is still to create streams once and reuse them, as driver overhead becomes a larger fraction of total time.

### Summary

| Pattern | Use streams? | RTX 4060 | GB10 |
|---|---|---|---|
| Single op, immediate sync | Irrelevant | no change | no change |
| Chain of N ops on the same data | Depends on what a sync costs | ~2.7× faster | ~1.1× |
| Independent ops on separate data | **Yes on GB10**, no on the 4060 | ~1× | up to ~5.2×, but non-monotonic in K |
| Compute while transferring data | **Yes — two streams** | 10–15% | ~4.6× |
| Stream creation per call | No — reuse streams | no overhead at scale | no overhead at scale |

The one portable lesson is that none of these rows is portable: measure on the target
machine before designing around any of them.

---

## Architecture

The data flow for a tensor operation (e.g. `a + b`):

```
Tensor<T>::operator+()
  → Tensor<T>::add(rhs, Stream::default_stream())          // tensor.inl
    → add_cpu(...)  or  launch_add(..., stream)            // ops/cpu/ or ops/kernels/
      → flat CPU loop  or  rank-specialized CUDA kernel
```

**`Tensor<T>`** — owning N-dimensional tensor. Stores shape, row-major strides, a raw `T*`, a `Device`, an `om::Stream`, and a `unique_ptr<Allocator<T>>`. Copy deep-copies via the allocator; move transfers ownership and nulls the source pointer.

**`Allocator<T>` / `AllocatorFactory<T>`** — abstract base with two implementations: `CpuAllocator` (malloc/free/memcpy) and `GpuAllocator` (cudaMalloc/cudaFree/cudaMemcpy). Selected at `Tensor` construction time from `DEVICE_TYPE`. The base declares the `*_async` entry points with **synchronous default implementations**, so a subclass overrides only what it can genuinely do asynchronously.

**`TensorView<T>`** — non-owning host-side view (raw pointer + shape/stride pointers + rank). Passed to CPU ops and converted to `DeviceTensorView` via `.as_device_tw()` before kernel launch.

**`DeviceTensorView<T>`** — non-owning device-side view. Shape and stride are stored as fixed inline arrays (`size_t shape[MAX_RANK]`, `size_t stride[MAX_RANK]`, `MAX_RANK = 8`) copied from host at construction — no device allocation. The struct is trivially copyable and passed by value to CUDA kernels, eliminating the 2×`cudaMalloc` + 2×`cudaFree` overhead that occurred on every kernel launch in earlier versions. Operator `()` is `__device__`-only.

**`om::Stream`** — RAII wrapper around `cudaStream_t`. The owning constructor calls `cudaStreamCreate`; `Stream(cudaStream_t)` wraps an existing handle without ownership. `Stream::default_stream()` returns a non-owning wrapper around `nullptr`, giving synchronous semantics without a code-path change. Every `Tensor<T>` stores an `om::Stream m_Stream`; the destructor calls `allocator->deallocate_async(ptr, m_Stream.get())` so memory is freed on the correct stream.

**Kernel dispatch (legacy)** — two macro families in `kernel_launcher.h`/`.inl`:
- `DEFINE_DEVICE_DISPATCH_BINARY_H` declares `op_dispatch<DEVICE_TYPE, T>` structs routing to `add_cpu` or `launch_add`.
- `DEFINE_DEVICE_DISPATCH_BINARY_INL` defines the free function `_add(…, DEVICE_TYPE)` that switches at runtime into the correct struct.

Since the stream refactor these are **mostly dead code**: the `_dispatch` structs take no `cudaStream_t`, so `tensor.inl` branches on `device_type()` itself and calls `add_cpu` / `launch_add` directly. Today only `Tensor::fill` still routes through the dispatch path. Adding a new op means wiring `tensor.inl` directly — register in `kernel_launcher` only if you also want the stream-less free function.

**Rank-specialized CUDA kernels** — `DEFINE_BINARY_OP_LAUNCH` generates a `launch_op` function that switches on `tensor.rank` (1–4) and selects a kernel with a rank-tuned grid/block layout. Rank ≥ 5 falls back to a flat 1D kernel (`_kernel_nd`) that reconstructs multi-indices from a linear index. Explicit template instantiations for `float`, `int`, `char`, `float16_t` are emitted per op.

**Contiguous fast path** — [headers/ops/kernels/contiguous.cuh](headers/ops/kernels/contiguous.cuh). Every elementwise launcher tries this first: since all tensors are contiguous row-major, the axis structure carries nothing the kernel needs, so the buffer is indexed linearly and every rank gets the rank-1 layout. Threads move `4 / sizeof(T)` elements each (1 for `float`/`int`, 2 for `float16_t`, 4 for `char`), packed through a 4-byte word. `TensorView::is_contiguous()` gates it, so the day a strided view exists it falls back to the rank-specialized kernels rather than reading the wrong elements. Covers `launch_add`/`sub`/`mul`/`div`, the scalar `_k` family, `launch_apply_op`, `launch_apply_binary_op` and `launch_fill`.

```
headers/ops/cpu/        ← CPU op declarations (macro-generated inline functions)
src/ops/cpu/            ← CPU op .cpp translation units
headers/ops/kernels/    ← CUDA kernel declarations and launch macros (.cuh)
src/ops/kernels/        ← CUDA kernel .cu translation units
```

---

## Design decisions

These are the non-obvious choices made during development, and why.

**Rank-specialized kernels over a single generic kernel**
A single flat kernel must reconstruct multi-dimensional indices from a linear offset at runtime, which adds per-thread division and modulo overhead and prevents rank-aware grid/block tuning. For rank 1–4, dedicated kernels use grid shapes matched to the tensor dimensions (e.g. a 2D `dim3(16,16)` block for rank-2), avoiding that overhead. Rank ≥ 5 falls back to `_kernel_nd`, which does the index reconstruction generically.

> **The benchmark contradicted this twice, and both are now fixed.** First the block
> sizes: the rank-1 launchers used `dim3(16)`, occupying a warp with half its lanes idle,
> so reshaping the same 16 M-element buffer to rank 2 made `add` go **149 → 187 GB/s**.
> Every rank-1 site now launches 256 threads.
>
> Then the premise itself. With the block sizes fixed, rank 1 was the *fastest* layout, not
> the slowest — 229.7 GB/s against 206 at rank 2, 141 at rank 3, 104 at rank 4 and 26 at
> rank 5. Mapping axes onto grid axes is what costs: only at rank 1 does a warp's 32 lanes
> stay contiguous in memory. And the indexing overhead the rank specialization exists to
> avoid is not worth paying for, because the shape is redundant — every tensor here is
> contiguous row-major, so the buffer can simply be indexed linearly. The contiguous fast
> path does that and brings every rank to 228–233 GB/s — re-measured end to end, all five
> ranks land within 1 % of each other. The rank-specialized kernels are still compiled and
> still correct; they are now the fallback for the strided views the library does not yet
> have.

**RAII for GPU memory via `Tensor<T>` + `Allocator<T>`**
Rather than pairing raw `cudaMalloc`/`cudaFree` calls at each use site, every `Tensor` owns a polymorphic `Allocator` chosen at construction time by `AllocatorFactory`. The destructor delegates to `allocator->deallocate`, making GPU memory lifetime deterministic regardless of exceptions or early returns — the same pattern used in PyTorch's `at::DataPtr`.

`CpuAllocator` deliberately stayed a thin `malloc`/`free` pair, on the reasoning that the
host path is not the interesting one. The benchmark priced that decision: above glibc's
128 KB mmap threshold every out-of-place op faulted in its entire output buffer from the
kernel, a 4.3× penalty on a 16 M-element `add` and 16× on a 64 MB device-to-host copy. A
caching allocator was not an optimization here, it was the difference between beating
NumPy and being 4× behind it — the A/B still costs 6.2× on a 16 M `add` when the cache is
turned off — so host allocations now go through `om::detail::HostPool`
([headers/host_pool.h](headers/host_pool.h)), a capped, mutex-guarded free list keyed by
size class, with a 64-byte header per block carrying its class (and giving 64-byte
alignment, where `malloc` guarantees 16). Same structure as NumPy's block cache and
PyTorch's CPU caching allocator, for the same reason.

**`DeviceTensorView` inline metadata instead of per-launch device allocations**
The original design allocated `shape[]` and `stride[]` in device memory on every `DeviceTensorView` construction (2×`cudaMalloc` + 2×`cudaMemcpy` per object; 6 allocations for a single binary op). Replacing those with fixed inline arrays (`size_t shape[MAX_RANK]`) eliminates all per-launch metadata allocations. The struct is now trivially copyable and passed by value into the kernel parameter block — the same pattern used by cuDNN and CUTLASS. `MAX_RANK = 8` covers practical use without wasting register space.

**Stream-aware allocator: `cudaMallocAsync` / `cudaFreeAsync`**
`GpuAllocator<T>` overrides `allocate_async` / `deallocate_async` with `cudaMallocAsync` / `cudaFreeAsync` (CUDA ≥ 11.2) so that tensors created on a non-null stream allocate and free memory without stalling the GPU. The base `Allocator<T>` provides sync fallbacks, so `CpuAllocator` and older CUDA versions work without changes. Every `Tensor<T>` stores the stream it was created on; the destructor frees on that same stream, ensuring the free is not issued before pending kernels finish.

**CUDA Streams as the canonical execution path**
All `Tensor<T>` methods have stream overloads (`tensor.add(rhs, stream)`). The no-stream variants are one-liner delegates to the stream version with `Stream::default_stream()` (a non-owning null stream wrapper), which gives synchronous behavior without duplicating any kernel dispatch logic. This makes the stream path the single source of truth and keeps the zero-stream user experience identical to the previous API.

**Runtime dispatch via macro-generated structs instead of virtual functions**
Using `virtual` dispatch for CPU vs. CUDA would add a vtable indirection on every element-wise op. Instead, `DEFINE_DEVICE_DISPATCH_BINARY_H` generates `op_dispatch<DEVICE_TYPE, T>` template specializations resolved at compile time. The only runtime branch is a `switch` on `DEVICE_TYPE` in the inlined free function, which the compiler can optimize away when the device is known statically.

---

## Key features

- **Rank-specialized kernels**: elementwise ops (add, sub, mul, div) with dedicated CUDA kernels for rank 1–4, each with a rank-tuned grid/block layout
- **N-dimensional support**: generic `_kernel_nd` fallback for rank ≥ 5 with stride-aware index reconstruction
- **Contiguous fast path**: elementwise launchers detect the (universal) contiguous case and index linearly, so rank 2–5 run at the same bandwidth as rank 1 — up to 8.9× on rank-5 `add`, with under 1.5 % spread across all five ranks
- **RAII GPU memory**: `Tensor<T>` owns a polymorphic `Allocator<T>` (CPU or GPU) with move semantics and no raw pointer leaks
- **Stream-aware allocator**: `GpuAllocator` uses `cudaMallocAsync`/`cudaFreeAsync`; each `Tensor` carries its stream and frees on it asynchronously
- **Zero-overhead stream API**: every op has a `(args, Stream&)` overload; no-stream variants delegate to `Stream::default_stream()` — one code path, two calling conventions
- **Inline `DeviceTensorView` metadata**: shape/stride stored as fixed arrays inside the view struct — eliminates 2×`cudaMalloc` + 2×`cudaFree` per kernel launch
- **Unified CPU/GPU API**: the same `operator+`, `operator-`, etc. work on both devices; dispatch is resolved at runtime from `DEVICE_TYPE`
- **Fused ops without intermediates**: functor composition (`Compose`, `BinaryCompose`) evaluated inside a single kernel — `relu`, `sigmoid`, `scale_shift`, `fused_add_mul`, …
- **Reductions**: `sum` / `mean` / `min` / `max` via a two-phase shared-memory tree plus warp shuffle (`__shfl_down_sync`)
- **Python package**: a ctypes binding over the C-ABI in `OpenMat.so`, exposing the same tensor and stream surface, with `__array_interface__` / `__cuda_array_interface__` for zero-copy interop

---

## Tensor API

```cpp
#include "tensor.cuh"
using om::Tensor;

auto a = Tensor<float>::ones({1024, 1024}, om::Device(0, om::DEVICE_TYPE::CUDA));
auto b = Tensor<float>::full({1024, 1024}, 2.0f, a.device());

auto c = a.matmul(b);            // 2D only
auto d = (a + b).relu();         // fused, no intermediate for the relu
float m = d.mean();              // reduction → host scalar
auto e = d.permute({1, 0}).reshape({1024 * 1024});
auto h = e.cpu();                // device → host copy
```

| Group | Methods |
|---|---|
| Factories | `zeros`, `ones`, `full`, `from_vector`, `fill` |
| Arithmetic | `add`, `sub`, `mul`, `div` + `+ - * /`, tensor–tensor and tensor–scalar |
| Linear algebra | `matmul` — **2D only**, no batching, no broadcasting |
| Reductions | `sum`, `mean`, `min`, `max` — synchronous, return a host scalar |
| Shape | `reshape`, `flatten`, `squeeze`, `unsqueeze` — **deep copies, not views** |
| Layout | `transpose` (**rank-2 only**, throws otherwise), `permute(axes)` |
| Fused | `apply`, `apply_binary`, `relu`, `sigmoid`, `scale_shift`, `shift_scale`, `fused_add_mul`, `fused_sub_mul`, `fused_mul_add`, `fused_div_add` |
| Transfer | `to(device)`, `cpu()`, `cuda()`, `copyToHost`, `copyToDevice` |

Every op above except the reductions and the host-side shape ops also has a
`(args, const Stream&)` overload.

**Dtypes.** `om::dtype<T>()` covers `float`, `double`, `int`, `char`, and `float16_t`
(a hand-rolled `__half` wrapper in `type_traits/types.cuh`). GPU kernels are instantiated
for `float`, `int`, `char`, `float16_t` — **`double` is CPU-only**. Generic code must gate
on `is_extended_arithmetic<T>` rather than `std::is_arithmetic`, or half precision is rejected.

**No aliasing views.** Unlike NumPy/PyTorch, nothing in this library returns a view onto
another tensor's buffer — `reshape` and friends allocate and copy, so a write through the
result never shows up in the original.

---

## Build

Requirements: NVIDIA GPU, CUDA Toolkit ≥ 11.2 (for `cudaMallocAsync`), CMake ≥ 3.24 (for
`CMAKE_CUDA_ARCHITECTURES=native`), a C++17/CUDA 17 compiler, OpenMP (`find_package(OpenMP REQUIRED)`;
bundled as `libgomp` with a stock GCC install, nothing extra to install on most systems).
Verified on CUDA 13.0 / GCC 13.3 / CMake 3.28 / GB10 (sm_121), all 14 suites passing
(11 correctness suites in 4.7 s, plus three timing/soak suites).

```bash
git clone https://github.com/AntonioPalese/OpenMat.git
cd OpenMat

# CMake only *warns* if this is unset — the library still builds, then every
# executable fails to link with `cannot find -lcudart`.
export CMAKE_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/$(uname -m)-linux-gnu"

./compile.sh          # clean rebuild + refreshes compile_commands.json
```

Or by hand:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j$(nproc)
```

Produces `build/OpenMat.so` (also what the Python package loads), `build/OpenMat_app`,
and `build/tests/test_*`.

Notes:
- `CMAKE_CUDA_ARCHITECTURES` defaults to `native`; override with `-DCMAKE_CUDA_ARCHITECTURES=<sm>`
  to cross-compile.
- The default `CMAKE_BUILD_TYPE` is `Debug`. Pass `-DCMAKE_BUILD_TYPE=Release` before timing anything.
- Sources are collected with `file(GLOB …)`, so a newly added `.cpp`/`.cu` needs a CMake re-run,
  not just `make`.

### Tests

GoogleTest, fetched by CMake via FetchContent; one binary per suite.

```bash
cd build && ctest                     # all 14 suites
./tests/test_arithmetic               # a single suite, per-test output
./tests/test_arithmetic --gtest_filter="TensorArithmetic.CPUOperations"
```

`test_arithmetic`, `test_fused_ops`, `test_device_transfer`, `test_factory`,
`test_reductions`, `test_reshape`, `test_transpose`, `test_streams`, `test_allocator_stream`,
`test_host_pool` and `test_contiguous` are correctness suites. `test_benchmarks`, `test_stress` and `test_stream_perf` are
timing/soak suites — slow, and meaningless in a Debug build.

---

## Python

The Python package is a **ctypes** binding (not pybind) over the C-ABI layer compiled into
`OpenMat.so`. See [python/README.md](python/README.md) for the full surface.

```bash
cd python
uv sync
uv pip install -e .
```

The loader looks for the library at `$OPENMAT_LIB`, then the copy bundled in the wheel, then
`<repo>/build/OpenMat.so` — so a source checkout works straight after `./compile.sh` with no
install step.

```python
import openmat as om
from openmat import Tensor, Stream

a = Tensor([[1.0, 2.0], [3.0, 4.0]])       # nested lists, flat lists or ndarrays
b = om.ones([2, 2])

c = (a @ b).relu()
print(c.tolist(), c.mean())

if om.cuda_is_available():
    with Stream() as s:
        g = a.cuda(stream=s)
        d = g.add(g, stream=s).sigmoid(stream=s)
        s.synchronize()
        print(d.cpu().tolist())            # or .numpy(), zero-copy via __array_interface__
```

- Dtypes exported to Python: **`float32` and `int32`** (`Tensor<double>` and `Tensor<char>` are not exported).
- Host tensors expose `__array_interface__`, CUDA tensors `__cuda_array_interface__`.
- Ops with a C++ stream overload take a `stream=None` keyword argument.
- Streams are **reference-counted on the C side**, not in Python: `cudaMallocAsync` memory must be
  freed on the stream that produced it, and Python's cyclic collector finalizes a cycle in arbitrary
  order. Each tensor holds one C-side reference, so `Stream.close()` is safe while its tensors are alive.

```bash
cd python && pytest        # test_tensor, test_tensor_api, test_dtypes, test_streams
```

---

## Roadmap

- [x] Rank-specialized 2D kernels (matmul, transpose, elementwise)
- [x] N-dimensional tensor with stride metadata
- [x] RAII GPU memory abstraction (`Tensor<T>` + `Allocator<T>`)
- [x] Runtime kernel dispatch by rank
- [x] CUDA Streams — full async API with `om::Stream` RAII wrapper
- [x] Stream-aware allocator (`cudaMallocAsync` / `cudaFreeAsync`)
- [x] Inline `DeviceTensorView` metadata (zero per-launch device allocations)
- [x] Transpose and N-D permute (CPU + GPU, tiled 2D kernel)
- [x] Fused ops (`scale_shift`, `fused_add_mul`, functor composition)
- [x] Reductions (`sum`, `mean`, `min`, `max` — shared-memory tree + warp shuffle)
- [x] `matmul` (2D, CPU + GPU)
- [x] Python bindings via C-ABI FFI layer (ctypes, float32 + int32, streams included)
- [x] `float16_t` support in kernels and `test_benchmarks`
- [x] Benchmark suite comparing against NumPy / PyTorch (`scripts/bench_vs.py`)
- [x] OpenMP parallelism for CPU `add`/`sub`/`mul`/`div` and `matmul` (size-gated `parallel for`)
- [x] `ikj` + L2 tiling + OpenMP for `matmul_cpu` — 1.81 → 123 GFLOP/s at 1024³
- [x] 8-lane accumulator for `reduce_sum_cpu` — 7.7 → 36.9 GB/s, now ahead of NumPy
- [x] Contiguous fast path for elementwise GPU kernels (every rank at rank-1 bandwidth)
- [ ] Same OpenMP treatment for `apply`/`apply_binary` (`relu`, `sigmoid`, `fused_*`)
- [ ] Threaded `sum` and a cross-thread `reduction(min:)/(max:)` for `min`/`max`
- [ ] cuBLAS integration as an optional matmul backend
- [ ] Random initialization (cuRAND)
- [ ] Broadcasting and batched matmul
- [ ] Aliasing views for `reshape` / `transpose`
- [ ] Mixed-precision support (BF16)
- [ ] Autograd prototype

Detailed planning lives in [docs/roadmap.md](docs/roadmap.md) (written in Italian), which
ends with a done/not-done priority table.

---

## What I learned

Building this from scratch exposed a set of problems that high-level frameworks abstract away entirely:

- **Memory coalescing is not free** — a naïve transpose kernel hits ~15% of peak memory bandwidth. A tiled implementation with shared memory and padding to avoid bank conflicts gets to ~85%.
- **Occupancy vs. shared memory is a real trade-off** — larger tiles improve reuse but reduce the number of resident warps. The sweet spot depends on the specific GPU's L1/shared memory ratio.
- **Stride bugs are silent** — a wrong stride in an N-D kernel produces numerically plausible but incorrect output. Systematic testing against NumPy reference output was the only reliable detection method.
- **Streams only help when you stop synchronizing — and how much they help is not portable** — the biggest win on the RTX 4060 (2.68×) came not from running kernels faster but from removing 99 out of 100 host/device sync barriers in a chain. Re-running the same suite on a GB10 returned 1.03× for that case, because a sync round-trip there costs almost nothing, while the parallel fan-out that was a wash on the 4060 became a 3.3× win. Same code, same test, opposite conclusions.
- **Raw arrays decay in CUDA kernel parameters** — passing `size_t axes[MAX_RANK]` as a kernel argument silently becomes a host pointer on the device side. The fix is a trivially-copyable struct wrapper so CUDA copies the data by value into the kernel parameter block.
- **A Python reference is not a lifetime guarantee** — the first two attempts at keeping a CUDA stream alive from Python both segfaulted under `gc.collect()`. The cyclic collector finalizes a cycle's members, plus everything reachable only from them, in arbitrary order, so a `Stream` could be torn down while tensors still owed it a stream-ordered free. Reference counting had to move to the C side of the boundary.
- **Half a warp is half the bandwidth** — the rank-1 elementwise kernels launched `dim3(16)` blocks, so 16 of 32 lanes in the warp sat idle. It cost 20–32% of memory bandwidth and went unnoticed for as long as there was nothing to compare against: the kernels *were* faster than the CPU path, which is all the internal benchmarks could tell me. It took reshaping the same buffer to a different rank, and watching `add` jump from 149 to 187 GB/s, to see it.
- **The optimization everyone recommends was the wrong one** — the obvious fix for an elementwise kernel is `__restrict__` pointers, a grid-stride loop and `float4` vector loads. Measured on the target GPU, the vector loads made `add` *slower* (216 GB/s against 235 for one scalar `float` per thread) and the grid-stride loop cost another 8–12 %: one 16-byte load per thread buys coalescing the hardware already had, and pays for it in the thread-level parallelism that keeps the memory pipeline full. What actually mattered was something the advice does not mention — that no thread move *less* than 4 bytes, which is why `char` sat at 193 GB/s — and something not about the kernel at all: that the launcher stop mapping tensor axes onto grid axes when the tensor is contiguous. The 8.9× at rank 5 came from deleting an indexing scheme, not from adding an instruction.
- **A stale benchmark number is worse than no number** — the table said GPU `add` ran at 159 GB/s, and that was the figure driving the next round of work. It had been true; a block-size fix landed afterwards and quietly moved it to 220 GB/s, but the report was not re-run, so the diagnosis it supported ("the indexing is too slow at rank 1") was aimed at a problem that no longer existed. Re-measuring before optimizing turned up the real one, two ranks over.
- **A benchmark is a measurement of the whole program, not the code you were looking at** — OpenMat's CPU `add` looked 4× slower than NumPy's at 16 M elements. The loop was not the problem; `malloc` was. Past glibc's 128 KB threshold every result buffer is a fresh `mmap`, and the op pays 16 384 page faults before it computes anything. One environment variable (`MALLOC_MMAP_MAX_=0`) closed the entire gap, and a size-classed free list in `CpuAllocator` made it permanent. The same cause was quietly making a 64 MB device-to-host copy look 24× slower than PyTorch's, in what appeared to be an unrelated part of the library.
- **Fusion is worth more than kernel tuning** — the ops where OpenMat beats PyTorch and NumPy are not the ones with the best kernels, they are the ones that avoid a memory round-trip. `fused_add_mul` wins by 1.66× against a mature library with better kernels, purely because `(a + b) * s` costs PyTorch a 64 MB intermediate. Arithmetic is nearly free at this scale; every avoided traversal of memory is not.
- **An OpenMP pragma is a request, not a guarantee — measure it like any other change.**
  `reduce_min_cpu`/`reduce_max_cpu` got a `#pragma omp simd reduction(min:)/(max:)`
  because it looked like the natural counterpart to the `#pragma omp parallel for` that
  sped up `add`/`sub`/`mul`/`div` elsewhere in the same pass. Isolated before/after timing
  — same loop body, same compiler flags, only the pragma differing — showed it 1.6×
  *slower*, not faster. `-fopt-info-vec-optimized` explained why: GCC 13 already
  auto-vectorizes that exact branch-and-select idiom under `-O3` alone, and the explicit
  reduction clause forced a worse lowering on top of a loop that was already vectorized.
  Reverted, and left as a comment in the code rather than a silent diff, so it does not
  get re-added on the assumption that it obviously must have helped.
- **The gaps called "structural" were one-line algorithm changes** — the report called CPU `matmul` (421× off NumPy) "the one structural gap rather than a tuning gap", and filed CPU `sum` (4× off) alongside it as a known-but-deferred cost. `matmul` needed a loop reorder from `ijk` to `ikj`, a tile size and an `omp parallel for`: 1.81 → 123 GFLOP/s, 68×. `sum` needed eight accumulators instead of one, so the loop runs at FP *throughput* instead of FP *latency*: 7.7 → 36.9 GB/s, from 4× behind NumPy to slightly ahead. Neither touched a kernel, an allocator or a dispatch path. "Structural" was a claim about how much work a fix would be, and it was wrong twice — worth distrusting the next time it appears in my own notes.
- **A benchmark harness measures itself too** — the cross-framework table reported a 64 MB device-to-host copy at 39.8 ms, which would have been a catastrophic regression. The tell was that PyTorch's own D2H degraded in the same run, 1.14 → 6.81 ms, on code neither I nor anyone else had touched. In a fresh process both libraries measure 1.136 ms. The transfer cases run last, after the process has accumulated two host caches, PyTorch's CUDA caching allocator and every live operand from the CUDA sweep, and on a unified-memory part that pressure lands on the transfer. The same contradiction had been sitting in the report for an edition — one section claiming 1141 µs for the op another section put at 28953 µs — and had been left unreconciled rather than treated as the signal it was.
- **`cudaMallocAsync` is not a drop-in replacement** — it uses a stream-ordered memory pool. Freeing on a different stream than the one used for allocation is a programming error that manifests as an illegal memory access with no obvious call site. Storing `m_Stream` in each `Tensor` and using it in the destructor is the invariant that keeps this safe.

---

## License

MIT