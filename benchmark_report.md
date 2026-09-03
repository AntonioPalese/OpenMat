# OpenMat vs NumPy vs PyTorch

Raw output of [scripts/bench_vs.py](scripts/bench_vs.py). Companion to
[stream_perf_report.md](stream_perf_report.md), which covers OpenMat's stream
overlap in isolation; this one places the library against the two references a
user would otherwise reach for.

## Environment

| | |
|---|---|
| Date | 2026-09-03 |
| GPU | NVIDIA GB10, `sm_121`, CUDA runtime 13.0 |
| CPU | 20× ARMv9 (Cortex-X925 + Cortex-A725), aarch64 |
| Kernel | Linux 6.17.0-1031-nvidia, glibc 2.39 |
| OpenMat | 0.2.0, **Release** build (`build-release/OpenMat.so`), `sm_121` native |
| NumPy | 2.5.2 |
| PyTorch | 2.14.0+cu130 (20 intraop threads) |
| dtype | `float32` throughout |

Method: per case, warm up, calibrate a repetition count so one batch lasts
≥ 50 ms, then take the **minimum** per-op time over 7 batches. CUDA cases
synchronise once per batch, identically for both libraries. Every op is
out-of-place and allocates its result, as all three libraries do by default.

## Headline

| | OpenMat vs best reference |
|---|---|
| GPU fused `(a+b)*s`, 16 M elem | **1.66× faster than PyTorch** (858 µs vs 1420 µs) |
| GPU elementwise `add`, 16 M elem | 1.47× slower (159 GB/s vs 234 GB/s) |
| GPU transpose, 2048² | **1.16× faster than PyTorch** (206 µs vs 240 µs) |
| CPU elementwise, 16 M elem, single-thread | parity with NumPy — 4.50 ms vs 6.13 ms on `add` (§3) |
| CPU elementwise (`add`/`sub`/`mul`/`div`), ≥1 M elem, OpenMP | **now faster than NumPy outright** — up to 10.6× over single-thread at 1M elem (§7) |
| CPU `min`/`max`, `omp simd reduction` | tried, **reverted** — measured 1.6× *slower*, not faster (§7) |
| CPU fused `x*s+t`, 16 M elem | **2.31× faster than NumPy, 1.24× faster than PyTorch** |
| `sum`, CPU | 4–12× slower (§5) |
| `matmul`, CPU 1024³ | 421× slower — 1.81 GFLOP/s vs 756 GFLOP/s (§6) |
| `matmul`, GPU 1024³ | 10.5× slower — 1.58 TFLOP/s vs 16.6 TFLOP/s (cuBLAS) |
| H2D / D2H transfer | parity, within 5 % |

## 1. Fusion is the win, and it is measurable

The library's thesis — one kernel instead of an intermediate buffer — holds at
scale on both backends.

At 16 M elements on the GPU, `a.fused_add_mul(b, 2.5)` runs in **858 µs** while
PyTorch's `(a + b) * 2.5` takes **1420 µs**: PyTorch materialises `a+b` into a
64 MB temporary and reads it back, OpenMat does not. OpenMat reaches 235 GB/s of
effective traffic on that op — the highest figure any op in this run achieved,
and essentially the machine's attainable ceiling.

On the CPU (allocator-corrected — since §3 that is simply the default) the same
effect shows twice:

| op, 16 M elem | NumPy | PyTorch | OpenMat |
|---|---|---|---|
| `(a+b)*s` | 6376 µs | 3376 µs | **4489 µs** |
| `x*s+t` | 5267 µs | 2831 µs | **2279 µs** |
| `relu` | 3474 µs | 1367 µs | 2288 µs |

`scale_shift` beats both references outright. NumPy loses here for the same
reason PyTorch loses on the GPU: `a * 2.0 + 1.0` is two passes over 64 MB.

## 2. GPU rank-1 kernels run at half occupancy

Every rank-1 launch except one uses a 16-thread block:

- [headers/ops/kernels/binary_op_macros.cuh:23](headers/ops/kernels/binary_op_macros.cuh#L23) — `dim3 threads(16)` for `add`/`sub`/`mul`/`div`
- [src/ops/kernels/fused_op.cu:79](src/ops/kernels/fused_op.cu#L79) — `dim3 threads(16)` for `apply_op_rank1` (`relu`, `sigmoid`, `scale_shift`, `shift_scale`)
- [src/ops/kernels/fused_op.cu:205](src/ops/kernels/fused_op.cu#L205) — `dim3 threads(256)` for `apply_binary_op_rank1` — **the exception**

A 16-thread block occupies one warp with 16 of 32 lanes idle. Measured on the
same 16 M-element buffer, reshaped so the launcher picks a different rank:

| shape | block | `add` | `relu` | `fused_add_mul` |
|---|---|---|---|---|
| rank 1 | 16 / 256\* | 149 GB/s | 118 GB/s | **218 GB/s**\* |
| rank 2 | 16×16 = 256 | **187 GB/s** | **174 GB/s** | 191 GB/s |
| rank 5 (`_nd`) | 256 | 164 GB/s | 169 GB/s | 157 GB/s |

\* `fused_add_mul` is the one op already launching 256 threads at rank 1, and
it is the only one at full bandwidth there.

Reshaping a vector to a matrix should not make elementwise addition 25 % faster.
Raising the two `threads(16)` sites to 256 is a two-line change.

## 3. The CPU gap above 128 KB was the allocator, not the loop — fixed

At 16 M elements OpenMat's CPU `add` measures 25.2 ms against NumPy's 6.1 ms —
a 4× gap that vanishes under `MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=-1`:

| 16 M elem, CPU | default glibc | mmap disabled |
|---|---|---|
| OpenMat `add` | 25210 µs | **4478 µs** |
| NumPy `add` | 6140 µs | 4402 µs |
| OpenMat `copy` | 21670 µs | **2740 µs** |
| OpenMat D2H, 64 MB | 27130 µs | **1130 µs** |
| PyTorch D2H, 64 MB | 1140 µs | 1140 µs |

`CpuAllocator` calls `malloc`/`free` directly. Above glibc's 128 KB threshold
every allocation is a fresh `mmap`, so each out-of-place op faults in its entire
output buffer from scratch — 16384 page faults per 64 MB result. NumPy and
PyTorch both cache host blocks and skip that cost. Corrected for it, OpenMat's
scalar loop lands within 2 % of NumPy on `add`.

This is also the whole of the 25× D2H anomaly: `Tensor::cpu()` allocates the
destination, and `cudaMemcpy` into never-touched pageable memory pays the faults
inside the copy. Corrected, it matches PyTorch to within 1 %.

**Fixed** in `CpuAllocator`, which now allocates through `om::detail::HostPool`
([headers/host_pool.h](headers/host_pool.h)): a process-wide free list keyed by
size class (8 classes per octave), capped at 256 MB, so a freed block is handed
back to the next allocation of its class with its pages still mapped instead of
being `munmap`'d. Re-measured on the same machine, same methodology, with the
cache on and with `OPENMAT_HOST_CACHE_BYTES=0` (which restores the old
behaviour) — PyTorch was not installed for this second run, so its column is the
figure from the run above:

| 16 M elem, CPU | cache off | cache on |
|---|---|---|
| OpenMat `add` | 19381 µs | **4498 µs** |
| NumPy `add` | 6024 µs | 6134 µs |
| OpenMat `copy` | 16366 µs | **2645 µs** |
| OpenMat D2H, 64 MB | 18239 µs | **1141 µs** |
| PyTorch D2H, 64 MB | — | 1140 µs (earlier run) |
| OpenMat H2D, 64 MB | 1191 µs | 1194 µs |

`add` lands at parity with NumPy and D2H within 0.1 % of PyTorch. The per-op
dispatch floor is unchanged (1.75 µs at 4096 elements, cache on and off): the
mutex and hash lookup do not register against the ctypes crossing.

H2D is the one number the cache does not move, and it is not supposed to: its
destination is device memory, and its source is pageable host memory, which
makes `cudaMemcpyAsync` synchronous.

**Implemented since:** `PinnedCpuAllocator` (`headers/allocator.h`) allocates
through `om::detail::PinnedHostPool` — `cudaHostAlloc` recycled the same way
`HostPool` recycles pageable blocks, with its own smaller cap
(`OPENMAT_PINNED_CACHE_BYTES`, default 64 MB; page-locking is one to two
orders of magnitude slower than a pageable allocation, so recycling matters
even more here). Two call sites use it without guessing which host tensors
will ever cross the bus: `Tensor::to()` allocates the destination of a
device-to-host copy pinned, because at that call site the buffer's only
purpose *is* to receive that exact copy; and `Tensor::pinned(shape)` lets a
caller pin a buffer explicitly when it knows it will be a repeated H2D
*source*, which `to()` cannot do retroactively for a tensor that already
exists. Neither path page-locks a tensor nobody asked to have pinned, and
neither is reachable without a working CUDA driver already in use, so the
CPU-only CI job is untouched (`PinnedHostPool` still has to compile there,
against the stub `libcuda.so` it links against, but nothing exercises it).

Measured on this machine (isolating the copy itself — same
`cudaMemcpyAsync` call, only the destination/source allocator differs — at
64 MB, 50 rounds, warmed):

| direction | pageable | pinned |
|---|---|---|
| D2H | 1.137 ms (59.0 GB/s) | 1.139 ms (58.9 GB/s) |
| H2D | 1.143 ms (58.7 GB/s) | 1.143 ms (58.7 GB/s) |

No measurable difference here — consistent with GB10 being the
"coherent-memory part" already noted in [README.md](README.md)'s stream
section: NVLink-C2C unified host/device memory means the driver has no
PCIe-bound bounce-buffer stage to skip by pinning, which is the mechanism
this was meant to remove. The fix is real and does what it says (verified:
`cudaPointerGetAttributes` reports `cudaMemoryTypeHost` on the allocated
block, and `test_host_pool`'s `PinnedHostPoolTensor` suite checks the
automatic and opt-in paths end to end), but on *this* GPU it is not the
lever — pinning should still matter on a discrete, PCIe-attached part like
the RTX 4060 in [stream_perf_report.md](stream_perf_report.md), where a
pageable-memory transfer really does route through the driver's internal
pinned staging buffer; that machine was not available to re-measure this
change against.

## 4. Small sizes are FFI-bound

Below ~1 M elements the measurement is dispatch overhead, not compute.

| 4096 elem | NumPy | PyTorch | OpenMat |
|---|---|---|---|
| CPU `add` | 0.44 µs | 0.79 µs | 1.69 µs |
| CUDA `add` | — | 3.22 µs | 8.69 µs |

OpenMat's floor is ~1.7 µs on CPU and ~8 µs on CUDA per op. Each call crosses
`ctypes` (argument boxing, a 512-byte error buffer allocation, a `Tensor._wrap`
with `om_stream_retain`), then heap-allocates a `Tensor<T>` on the C++ side.
PyTorch's ~3 µs CUDA floor is a compiled dispatcher over a caching allocator.
This is the cost of the ctypes binding choice and is a fixed constant — it is
already amortised away by 1 M elements.

## 5. `sum` is a serial scalar accumulate

`reduce_sum_cpu` ([headers/ops/cpu/reduce_cpu.h:9](headers/ops/cpu/reduce_cpu.h#L9))
is `for (i) acc = acc + src[i]` into a single accumulator. The loop-carried
floating-point dependency blocks auto-vectorisation without `-ffast-math`, so the
loop runs at one add per FP latency: **7.7 GB/s**, against ~93 GB/s for NumPy's
pairwise SIMD reduction. The gap is 4× at 16 M and 54× at 1 M — and unlike §3 it
is not allocator noise, since the op returns a host scalar and allocates nothing.

Four or eight independent partial accumulators reduced at the end would recover
most of it. The GPU reduction is fine: 320 µs vs PyTorch's 278 µs at 16 M.

## 6. `matmul` is the one structural gap

`matmul_cpu` ([headers/ops/cpu/matmul_cpu.h](headers/ops/cpu/matmul_cpu.h)) is the
textbook `ijk` triple loop, indexing `rhs(k, j)` down a column — a cache miss per
inner iteration — with no blocking, no SIMD and no threading, against OpenBLAS.

| 1024³ | GFLOP/s |
|---|---|
| OpenMat CPU | 1.81 |
| NumPy CPU | 756 |
| PyTorch CPU | 762 |
| OpenMat CUDA | 1582 |
| PyTorch CUDA (cuBLAS) | 16592 |

The GPU kernel ([src/ops/kernels/matmul_gpu.cu:14-61](src/ops/kernels/matmul_gpu.cu#L14-L61))
is already tiled — it stages 16×16 tiles of A and B in shared memory with the
usual two `__syncthreads()` per tile. The 10.5× is what is left *after* that, and
it comes from three things the kernel does not do:

- **One output element per thread — no register blocking.** The inner loop is
  `sum += tileA[ty][k] * tileB[k][tx]`: two shared-memory loads per FMA. The LDS
  issue rate, not the FFMA rate, is the ceiling, and no amount of occupancy moves
  it. Having each thread compute a 4×4 patch in registers turns 32 loads per 16
  FMAs into 8 per 16 — 4× less shared traffic per FLOP — and is the single
  largest remaining win.
- **No double buffering.** The two barriers per iteration make the loop strictly
  load → sync → compute → sync, so every block eats the full global-memory
  latency of tile *t* with nothing in flight to hide it. Prefetching tile *t+1*
  into a second shared buffer (`cp.async` on sm_80+) overlaps the two.
- **No tensor cores.** Every product is a scalar FFMA on the CUDA cores; nothing
  goes through `wmma`/`mma.sync`, where the bulk of an sm_121 part's matmul
  throughput lives.

Global loads are scalar as well — one `T` per thread per tile, strided by
`strideA1` — rather than vectorised; a smaller win, but a free one once the tile
shape allows it.

The CPU path at 421× off is not a tuning gap; it is a different algorithm class.
Loop reordering to `ikj` alone would win roughly an order of magnitude for one
line of change.

## 7. CPU elementwise ops were single-threaded — OpenMP closes most of it

Everything above this point predates a separate pass: `DEFINE_BINARY_OPS_CPU`
and `DEFINE_UNARY_OPS_CPU` — the CPU side of `add`/`sub`/`mul`/`div`, tensor⊕tensor
and tensor⊕scalar, in [headers/ops/cpu/binary_op_macros.h](headers/ops/cpu/binary_op_macros.h)
and [headers/ops/cpu/unary_op_macros.h](headers/ops/cpu/unary_op_macros.h) — were a
single-thread scalar `for` loop, full stop. PyTorch on the same machine uses 20.
Since both macros are a single point of generation, one `#pragma omp parallel for
schedule(static) if(_total > 65536)` in each covers every op they generate at once.

**Methodology note:** this section is *not* a `scripts/bench_vs.py` run — measured
with a standalone before/after harness instead (same Release binary, same code
path, `OMP_NUM_THREADS` toggled between invocations as the single/multi-thread
proxy), NumPy 2.5.0, no PyTorch installed in this pass. Same machine and date as
the rest of this report (see Environment above).

### 7.1 `add`/`sub`/`mul`/`div` — the threshold does its job

`OMP_NUM_THREADS=1` vs unset (20 threads), same binary:

| op | N | 1 thread | 20 threads | speedup |
|---|---|---|---|---|
| `add` (tensor) | 4096 | 0.0020 ms | 0.0021 ms | 1.0× (below threshold — untouched) |
| `add` (tensor) | 65536 | 0.0065 ms | 0.0065 ms | 1.0× (at the threshold — `>`, not `≥`, still scalar) |
| `add` (tensor) | 1 048 576 | 0.110 ms | 0.016 ms | **6.9×** |
| `add` (tensor) | 16 777 216 | 4.16 ms | 2.01 ms | **2.1×** |
| `div` (tensor) | 1 048 576 | 0.270 ms | 0.026 ms | **10.6×** |
| `div` (tensor) | 16 777 216 | 4.35 ms | 2.01 ms | **2.2×** |
| `add` (scalar) | 1 048 576 | 0.070 ms | 0.013 ms | **5.2×** |
| `add` (scalar) | 16 777 216 | 2.35 ms | 1.36 ms | **1.7×** |

`sub`/`mul` track `add`; `mul` (scalar) tracks `add` (scalar) — omitted for space, same
shape. Below 65536 elements the `if()` clause means fork/join is never paid: every
row there sits within ~2% of the single-thread number. Above it, the win ranges
1.7–10.6× depending on op and size, largest exactly where the pre-existing bottleneck
(§3's allocator fix) had already removed the memory-fault tax and left the scalar
loop as the ceiling.

Against NumPy, on the same 20-thread run:

| op | N | OpenMat | NumPy | OpenMat vs NumPy |
|---|---|---|---|---|
| `add` (tensor) | 1 048 576 | 0.016 ms | 0.144 ms | **9.1× faster** |
| `div` (tensor) | 1 048 576 | 0.026 ms | 0.273 ms | **10.7× faster** |
| `add` (tensor) | 16 777 216 | 2.01 ms | 5.75 ms | **2.9× faster** |
| `div` (tensor) | 16 777 216 | 2.01 ms | 6.16 ms | **3.1× faster** |

At ≥1M elements `add`/`sub`/`mul`/`div` now beat NumPy outright instead of merely
reaching parity, which is what §3 alone bought.

### 7.2 `min`/`max` — the same trick, tried and reverted

The natural next step looked like `#pragma omp simd reduction(min:acc)` /
`reduction(max:acc)` on `reduce_min_cpu`/`reduce_max_cpu`
([headers/ops/cpu/reduce_cpu.h](headers/ops/cpu/reduce_cpu.h)): too little work per
element for a thread team, but exactly the branch-and-select idiom the clause is
meant to recognize. Measured, it made things worse.

Isolated A/B — identical loop body, only the pragma differs, both compiled
`-O3 -march=native -fopenmp` in the same translation unit so it is the pragma's
effect being measured and not a flags difference:

```
N              before(ms)      after(ms)  speedup
4096              0.00150        0.00290    0.52x
65536             0.02341        0.04667    0.50x
1048576           0.42398        0.74877    0.57x
16777216          4.81557        7.58425    0.63x
```

Reproduced with call order swapped (`after` timed first, `before` second) to rule
out a warm-cache bias — same ~0.62× result across 15 trials, three separate runs.
`-fopt-info-vec-optimized` explains it: **both** loops report `optimized: loop
vectorized using 16 byte vectors` — GCC 13 already auto-vectorizes the plain
`if (x < acc) acc = x` idiom under `-O3` alone, and the explicit `reduction(min:)`
clause forces a different, less efficient lowering on top of a loop that was
already vectorized, rather than improving on it.

Reverted. `reduce_min_cpu`/`reduce_max_cpu` are back to the plain scalar loop, and
`min`/`max` measure identically under `OMP_NUM_THREADS=1` and 20 threads (no
`parallel for` was ever added there — a single compare-and-select per element is
too little work for fork/join to pay for itself):

| op | N | 1 thread | 20 threads |
|---|---|---|---|
| `min` | 16 777 216 | 4.754 ms | 4.745 ms |
| `max` | 16 777 216 | 4.754 ms | 4.749 ms |

Still 2.7–6.7× slower than NumPy's `min`/`max` across the sizes measured — a real
gap, but a pre-existing one this pass did not create and did not close. `reduce_sum_cpu`
was not touched here: its earlier fix (8 independent accumulator lanes, breaking the
loop-carried FP dependency chain) is a source-level restructuring, not a pragma, and
is unaffected by any of the above.

`apply`/`apply_binary` (the CPU path behind `relu`, `sigmoid`, `scale_shift`,
`shift_scale`, and the four `fused_*` methods — [docs/fused_operations.md](docs/fused_operations.md))
were not part of this pass and remain single-threaded on CPU regardless of size;
the same `if(_total > N)` treatment as §7.1 would apply there unchanged.

## Full results

### CPU, default threading

```
op                             shape      np us  torch us     om us
elementwise/add              (4096,)       0.44      0.79      1.69
elementwise/mul              (4096,)       0.43      0.74      1.71
fused/(a+b)*s                (4096,)       0.96      2.16      1.81
fused/x*s+t                  (4096,)       1.01      2.66      1.84
unary/relu                   (4096,)       1.18      0.70      1.51
reduction/sum                (4096,)       0.62      0.99      2.73
elementwise/add             (65536,)       4.46      9.92      5.93
elementwise/mul             (65536,)       4.46      5.59      5.92
fused/(a+b)*s               (65536,)       7.35     25.03      6.11
fused/x*s+t                 (65536,)       5.46     19.19      5.90
unary/relu                  (65536,)      13.27      4.70      5.53
reduction/sum               (65536,)       6.20      5.80     34.44
elementwise/add           (1048576,)     143.97     16.00    149.79
elementwise/mul           (1048576,)     142.33     20.65    111.71
fused/(a+b)*s             (1048576,)    1198.10     42.01    112.94
fused/x*s+t               (1048576,)     823.21     33.26     68.15
unary/relu                (1048576,)     201.49     13.76     80.03
reduction/sum             (1048576,)     100.12     10.17    540.00
elementwise/add          (16777216,)    6139.90   1998.81  25210.45
elementwise/mul          (16777216,)    5691.64   1993.60  20209.29
fused/(a+b)*s            (16777216,)    9929.49   3379.99  19131.11
fused/x*s+t              (16777216,)    8253.64   2781.28  14573.71
unary/relu               (16777216,)    5327.44   1331.85  22245.16
reduction/sum            (16777216,)    2144.31    784.76   8820.19
linalg/matmul             (128, 128)      22.71     23.34    777.03
linalg/matmul             (512, 512)     431.76    400.60  72413.04
linalg/matmul           (1024, 1024)    2839.72   2816.86 1186536.09
shape/transpose           (512, 512)     152.73    110.81    177.24
shape/transpose         (2048, 2048)    5801.80   2711.69   6716.35
```

### CPU, `MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=-1`

Isolates the compute loop from the page-fault cost of §3. Taken before the block
cache landed; the default build now reaches these numbers on its own, and this
env-var pass is only how the diagnosis was made.

```
op                             shape      np us  torch us     om us
elementwise/add             (65536,)       4.52     10.84      5.88
fused/(a+b)*s               (65536,)       7.32     17.48      6.02
unary/relu                  (65536,)      13.26      7.41      5.49
elementwise/add           (1048576,)     106.28     15.80    153.27
fused/(a+b)*s             (1048576,)     285.70     48.08    150.22
fused/x*s+t               (1048576,)     198.20     22.65     81.69
unary/relu                (1048576,)     208.15     13.25     79.93
elementwise/add          (16777216,)    4401.67   1990.19   4477.57
elementwise/mul          (16777216,)    4399.53   2000.95   4118.35
fused/(a+b)*s            (16777216,)    6376.34   3375.52   4488.95
fused/x*s+t              (16777216,)    5267.17   2831.46   2279.16
unary/relu               (16777216,)    3473.65   1366.76   2287.97
reduction/sum            (16777216,)    2276.84    722.07   8770.59
```

### CUDA

```
op                             shape   torch us     om us   om GB/s  torch GB/s
elementwise/add              (4096,)      3.22      8.69       5.7        15.2
elementwise/mul              (4096,)      3.24      8.69       5.7        15.2
fused/(a+b)*s                (4096,)      7.08      8.72       5.6         6.9
fused/x*s+t                  (4096,)      7.26      8.60       3.8         4.5
unary/relu                   (4096,)      3.57      8.11       4.0         9.2
reduction/sum                (4096,)     11.25     13.18       1.2         1.5
elementwise/add             (65536,)      3.26     12.40      63.4       241.0
elementwise/mul             (65536,)      3.28     12.39      63.5       239.9
fused/(a+b)*s               (65536,)      7.21      9.34      84.2       109.0
fused/x*s+t                 (65536,)      7.30     11.74      44.7        71.9
unary/relu                  (65536,)      3.65     10.99      47.7       143.5
reduction/sum               (65536,)     21.51     14.05      18.7        12.2
elementwise/add           (1048576,)      8.20     73.99     170.1      1534.5
elementwise/mul           (1048576,)      8.20     73.98     170.1      1534.6
fused/(a+b)*s             (1048576,)     18.43     20.52     613.3       682.7
fused/x*s+t               (1048576,)     16.39     64.83     129.4       511.7
unary/relu                (1048576,)      8.19     62.04     135.2      1024.2
reduction/sum             (1048576,)     15.84     28.17     148.9       264.7
elementwise/add          (16777216,)    859.93   1262.97     159.4       234.1
elementwise/mul          (16777216,)    859.32   1263.72     159.3       234.3
fused/(a+b)*s            (16777216,)   1419.65    857.55     234.8       141.8
fused/x*s+t              (16777216,)   1122.70   1147.48     117.0       119.5
unary/relu               (16777216,)    563.14   1068.83     125.6       238.3
reduction/sum            (16777216,)    278.45    319.81     209.8       241.0
linalg/matmul             (128, 128)     10.23     13.40
linalg/matmul             (512, 512)     29.31    184.98
linalg/matmul           (1024, 1024)    129.43   1357.44
shape/transpose           (512, 512)     10.25     19.27     108.8       204.6
shape/transpose         (2048, 2048)    240.24    206.22     162.7       139.7
transfer/h2d                (65536,)     13.20     13.34      19.6        19.9
transfer/d2h                (65536,)     11.87     11.99      21.9        22.1
transfer/h2d              (1048576,)     79.49     80.56      52.1        52.8
transfer/d2h              (1048576,)     79.50     80.23      52.3        52.8
transfer/h2d             (16777216,)   1130.52   1184.69      56.6        59.4
transfer/d2h             (16777216,)   1139.89  28952.80       2.3        58.9
```

Note: sub-1 M GB/s figures are dispatch-bound, not bandwidth measurements; the
1 M CPU column reflects PyTorch running 20 threads over a cache-resident buffer.

## Reproducing

```bash
# Release build is mandatory — Debug numbers are meaningless.
mkdir build-release && cd build-release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. && make -j$(nproc)
cd ..

# numpy + torch in a throwaway env; the repo's own package needs neither.
uv venv bench-env
VIRTUAL_ENV=bench-env uv pip install numpy
VIRTUAL_ENV=bench-env uv pip install torch --index-url https://download.pytorch.org/whl/cu130

OPENMAT_LIB=build-release/OpenMat.so PYTHONPATH=python \
  bench-env/bin/python scripts/bench_vs.py --out bench_results.json

# same pass with the host block cache off, i.e. the pre-§3 allocator
OPENMAT_HOST_CACHE_BYTES=0 \
OPENMAT_LIB=build-release/OpenMat.so PYTHONPATH=python \
  bench-env/bin/python scripts/bench_vs.py --no-cuda --out bench_nocache.json
```

`--quick` shortens the batches for a smoke run. `bench_results.json` carries the
full environment block plus min/median/reps per case.
