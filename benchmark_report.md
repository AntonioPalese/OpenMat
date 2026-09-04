# OpenMat vs NumPy vs PyTorch

Raw output of [scripts/bench_vs.py](scripts/bench_vs.py). Companion to
[stream_perf_report.md](stream_perf_report.md), which covers OpenMat's stream
overlap in isolation; this one places the library against the two references a
user would otherwise reach for.

## Environment

| | |
|---|---|
| Date | 2026-09-04 |
| GPU | NVIDIA GB10, `sm_121`, CUDA runtime 13.0, driver 580.173.02 |
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

All 14 test suites pass against this build (11 correctness suites in 4.7 s,
plus the three timing/soak suites).

## Headline

| | OpenMat vs best reference |
|---|---|
| GPU elementwise `add`, 16 M elem | **1.02× faster than PyTorch** — 236.4 GB/s vs 232.4, at **any rank** (§8) |
| GPU fused `(a+b)*s`, 16 M elem | **1.68× faster than PyTorch** (851 µs vs 1428 µs) |
| GPU fused `x*s+t`, 16 M elem | **1.97× faster than PyTorch** (573 µs vs 1130 µs) |
| GPU `relu`, 16 M elem | parity — 573.2 µs vs 569.5 µs |
| GPU `transpose`, 2048² | **1.15× faster than PyTorch** (205 µs vs 236 µs) |
| CPU elementwise `add`, 16 M elem | **2.83× faster than NumPy**, parity with PyTorch (§7) |
| CPU fused `x*s+t`, 16 M elem | **3.06× faster than NumPy, 1.07× faster than PyTorch** |
| CPU `sum`, 16 M elem | **1.06× faster than NumPy** — the §5 gap is closed |
| CPU `matmul`, 1024³ | 6.3× slower — 123 GFLOP/s vs 768 (was 421× slower; §6) |
| H2D / D2H transfer | parity, within 5 % (§3) |
| GPU `sum`, 16 M elem | 1.16× slower — 317 µs vs 273 µs |
| CPU `min`/`max`, 16 M elem | 2.74× slower than NumPy (§7.2) |
| GPU `matmul`, 1024³ | 10.3× slower — 1.64 TFLOP/s vs 16.9 TFLOP/s (cuBLAS) |
| Per-op dispatch floor | 1.9 µs CPU, 7.9 µs CUDA (PyTorch: 0.8 / 3.1 µs) (§4) |

Since the previous run, four of these rows changed sign. GPU `add` went from
1.47× slower to marginally faster, GPU `x*s+t` and `relu` from ~2× slower to
1.97× faster and parity (§8, the contiguous fast path); CPU `sum` from 4×
slower to faster than NumPy (§5); and CPU `matmul` from 421× to 6.3× (§6).

## 1. Fusion is the win, and it is measurable

The library's thesis — one kernel instead of an intermediate buffer — holds at
scale on both backends.

At 16 M elements on the GPU, `a.fused_add_mul(b, 2.5)` runs in **851 µs** while
PyTorch's `(a + b) * 2.5` takes **1428 µs**: PyTorch materialises `a+b` into a
64 MB temporary and reads it back, OpenMat does not. `scale_shift` shows the
same effect more sharply still — **573 µs against 1130 µs, 1.97×** — because
`x * 2.0 + 1.0` costs PyTorch two full passes over the buffer where OpenMat
makes one.

Both fused ops now run at 234–237 GB/s, which is also what plain `add` reaches
(§8). That is the point: the fused op moves strictly less memory than the
unfused pair, and since these kernels are bandwidth-bound, less traffic is the
whole speedup.

On the CPU the same effect shows up at 16 M elements:

| op, 16 M elem | NumPy | PyTorch | OpenMat |
|---|---|---|---|
| `(a+b)*s` | 10488 µs | **3407 µs** | 4155 µs |
| `x*s+t` | 8119 µs | 2832 µs | **2655 µs** |
| `relu` | 5513 µs | **1393 µs** | 2298 µs |

`scale_shift` beats both references outright, and it does so while still
single-threaded (§7 parallelised `add`/`sub`/`mul`/`div`, not the fused family)
— NumPy loses to a one-threaded loop purely on avoided memory traffic.

## 2. GPU rank-1 kernels ran at half occupancy — fixed

*Historical; kept because §8 builds on it.* Every rank-1 launch except
`apply_binary_op_rank1` used a `dim3(16)` block, occupying one warp with 16 of
its 32 lanes idle. Measured on a 16 M-element buffer reshaped so the launcher
picked a different rank:

| shape | block | `add` | `relu` | `fused_add_mul` |
|---|---|---|---|---|
| rank 1 | 16 / 256\* | 149 GB/s | 118 GB/s | **218 GB/s**\* |
| rank 2 | 16×16 = 256 | **187 GB/s** | **174 GB/s** | 191 GB/s |
| rank 5 (`_nd`) | 256 | 164 GB/s | 169 GB/s | 157 GB/s |

\* `fused_add_mul` was the one op already launching 256 threads at rank 1, and
the only one at full bandwidth there.

Reshaping a vector to a matrix should not make elementwise addition 25 % faster.
Both `threads(16)` sites were raised to 256, which fixed rank 1 — and then §8
found the larger hole one rank over.

## 3. The CPU gap above 128 KB was the allocator, not the loop — fixed

At 16 M elements OpenMat's CPU `add` once measured 25.2 ms against NumPy's
6.1 ms — a 4× gap that vanished under `MALLOC_MMAP_MAX_=0`. `CpuAllocator`
called `malloc`/`free` directly, and above glibc's 128 KB threshold every
allocation is a fresh `mmap`, so each out-of-place op faulted in its entire
output buffer from scratch: 16384 page faults per 64 MB result. NumPy and
PyTorch both cache host blocks and skip that cost.

**Fixed** in `CpuAllocator`, which allocates through `om::detail::HostPool`
([headers/host_pool.h](headers/host_pool.h)): a process-wide free list keyed by
size class (8 classes per octave), capped at 256 MB, so a freed block is handed
back to the next allocation of its class with its pages still mapped instead of
being `munmap`'d.

The A/B still reproduces on this build — `OPENMAT_HOST_CACHE_BYTES=0` restores
the old behaviour, and it is worth **6.2×** on `add`:

| 16 M elem, CPU | cache off | cache on |
|---|---|---|
| OpenMat `add` | 12271 µs | **1990 µs** |
| OpenMat `mul` | 13071 µs | **1938 µs** |
| OpenMat `(a+b)*s` | 22430 µs | **4155 µs** |
| OpenMat `x*s+t` | 22050 µs | **2655 µs** |
| OpenMat `relu` | 14413 µs | **2298 µs** |
| OpenMat `sum` | 2120 µs | 1821 µs |
| NumPy `add` | 6252 µs | 5637 µs |

`sum` is the control: it returns a host scalar and allocates nothing, so the
cache moves it only within noise. Everything that allocates a 64 MB result
moves by 5–8×.

### Pinned host memory: implemented, and not the lever on this GPU

`PinnedCpuAllocator` (`headers/allocator.h`) allocates through
`om::detail::PinnedHostPool` — `cudaHostAlloc` recycled the same way `HostPool`
recycles pageable blocks, with its own smaller cap
(`OPENMAT_PINNED_CACHE_BYTES`, default 64 MB; page-locking is one to two orders
of magnitude slower than a pageable allocation, so recycling matters even more
here). Two call sites use it without guessing which host tensors will ever
cross the bus: `Tensor::to()` allocates the destination of a device-to-host
copy pinned, because at that call site the buffer's only purpose *is* to
receive that exact copy; and `Tensor::pinned(shape)` lets a caller pin a buffer
explicitly when it knows it will be a repeated H2D *source*.

Measured on this machine, 64 MB, warmed, isolating the copy itself:

| direction | pageable | pinned |
|---|---|---|
| D2H | 1.137 ms (59.0 GB/s) | 1.136 ms (59.1 GB/s) |
| H2D | 1.143 ms (58.7 GB/s) | 1.143 ms (58.7 GB/s) |

No measurable difference — consistent with GB10 being a coherent-memory part:
NVLink-C2C unified host/device memory means the driver has no PCIe-bound
bounce-buffer stage to skip by pinning, which is the mechanism this was meant
to remove. The fix is real and does what it says (`cudaPointerGetAttributes`
reports `cudaMemoryTypeHost` on the block, and `test_host_pool`'s
`PinnedHostPoolTensor` suite checks both paths end to end), but on *this* GPU it
is not the lever. It should still matter on a discrete, PCIe-attached part.

Where the allocator work *did* pay off on transfers is the repeated-round-trip
case: `Stress.AsyncTransferBandwidth` (100 × 64 MB H2D+D2H, allocating its
destination every round) went from **3109 ms / 4.3 GB/s to 263 ms / 51.1 GB/s**,
an 11.8× improvement. That number was called out as the outstanding bottleneck
in both this report and [stream_perf_report.md](stream_perf_report.md); it is
now closed.

### The `transfer/d2h` row at 16 M in the full table is an artifact — ignore it

The full CUDA table below reports `transfer/d2h (16777216,)` at 39780 µs for
OpenMat and 6815 µs for PyTorch. **Neither number is a measurement of the
copy.** Re-run in a fresh process, on the same build and the same buffers:

| 64 MB D2H, isolated | time | bandwidth |
|---|---|---|
| OpenMat `.cpu()` | 1.136 ms | 59.1 GB/s |
| PyTorch `.cpu()` | 1.136 ms | 59.1 GB/s |

Identical, and both matching the pinned/pageable table above. The tell is that
PyTorch degrades alongside OpenMat in the full run (its own D2H went 1.14 →
6.81 ms between runs) — a harness effect, not a library one. `transfer_bench`
runs last, after the process has accumulated the host block cache, the pinned
pool, PyTorch's CUDA caching allocator and every live operand from the CUDA
sweep; on a unified-memory part that memory pressure lands on the transfer.

This also resolves a contradiction that was already present in the previous
edition of this report, where §3 claimed 1141 µs for the same op the full table
put at 28953 µs. The isolated figure was the right one; the full-table row
should never have been quoted as a transfer result. The 64 KB and 1 M rows,
which run before the pressure builds, are at parity and are trustworthy.

## 4. Small sizes are FFI-bound

Below ~1 M elements the measurement is dispatch overhead, not compute.

| 4096 elem | NumPy | PyTorch | OpenMat |
|---|---|---|---|
| CPU `add` | 0.44 µs | 0.77 µs | 1.93 µs |
| CUDA `add` | — | 3.11 µs | 7.94 µs |

OpenMat's floor is ~1.9 µs on CPU and ~7.9 µs on CUDA per op. Each call crosses
`ctypes` (argument boxing, a 512-byte error buffer allocation, a `Tensor._wrap`
with `om_stream_retain`), then heap-allocates a `Tensor<T>` on the C++ side.
PyTorch's ~3 µs CUDA floor is a compiled dispatcher over a caching allocator.
This is the cost of the ctypes binding choice and is a fixed constant — it is
already amortised away by 1 M elements.

One floor did move: `sum` at 4096 elements went from 2.73 µs to **0.91 µs**,
now *faster* than PyTorch's 0.97 µs, because §5 removed the serial accumulate
that dominated even at that size.

## 5. `sum` was a serial scalar accumulate — fixed

`reduce_sum_cpu` ([headers/ops/cpu/reduce_cpu.h](headers/ops/cpu/reduce_cpu.h))
used to be `for (i) acc = acc + src[i]` into a single accumulator. The
loop-carried floating-point dependency blocks auto-vectorisation without
`-ffast-math`, so the loop ran at one add per FP *latency* rather than per
throughput: 7.7 GB/s, against NumPy's pairwise SIMD reduction.

The fix is a source-level restructuring, not a pragma: the accumulation is
split across **8 independent lanes** reduced pairwise at the end, which breaks
the dependency chain and lets the compiler vectorise and pipeline them. It is
also slightly *more* accurate than one long straight-line accumulation, since
the partials merged at the end are of similar magnitude.

| `sum`, CPU | NumPy | PyTorch | OpenMat |
|---|---|---|---|
| 4096 | 0.61 µs | 0.97 µs | **0.91 µs** |
| 65536 | 6.20 µs | 6.33 µs | **4.82 µs** |
| 1 048 576 | 100.80 µs | **11.37 µs** | 66.67 µs |
| 16 777 216 | 1929 µs | **650 µs** | **1821 µs** |

At 16 M, **36.9 GB/s against NumPy's 34.8** — OpenMat is now 1.06× faster than
the like-for-like single-threaded reference, where it was 4× slower. PyTorch
still wins by 2.8× at the two large sizes because it threads the reduction
across 20 cores; `reduce_sum_cpu` is deliberately still single-threaded, and a
`parallel for` with a `reduction(+:)` clause is the obvious next step there.

The GPU reduction was never the problem and is unchanged: 317 µs vs PyTorch's
273 µs at 16 M.

## 6. `matmul`: the CPU gap closed 68×, the GPU one remains

`matmul_cpu` ([headers/ops/cpu/matmul_cpu.h](headers/ops/cpu/matmul_cpu.h)) was
the textbook `ijk` triple loop, indexing `rhs(k, j)` down a column — one cache
miss per inner iteration — with no blocking, no SIMD and no threading, against
OpenBLAS. It measured **1.81 GFLOP/s** at 1024³, 421× off NumPy.

It is now `ikj` order with 128-wide L2 tiling and an `omp parallel for` over the
independent output rows. Walking `k` in the middle makes both the `rhs` row and
the `dst` row sweep contiguously in the innermost `j` loop — unit stride, so the
compiler vectorises it — and blocking `i`/`k`/`j` keeps each panel L2-resident
across the accumulation.

| CPU matmul | OpenMat GFLOP/s | NumPy GFLOP/s | ratio |
|---|---|---|---|
| 128³ | **196.6** | 189.2 | **1.04× faster** |
| 512³ | 140.6 | 666.8 | 4.7× slower |
| 1024³ | 122.8 | 768.3 | 6.3× slower |

68× faster than the `ijk` loop at 1024³, and at 128³ it now edges out NumPy,
where the call is small enough that OpenBLAS' own dispatch overhead is
comparable to the work. What remains at 512³ and above is the gap between a
tiled C loop and a hand-tuned microkernel: OpenBLAS blocks for L1 *and*
registers, emits NEON intrinsics with software pipelining, and packs its panels
into contiguous scratch buffers so the innermost loop streams. That is a
different implementation class, not a tuning delta — but it is now a 6× gap
rather than three orders of magnitude.

On the GPU, unchanged:

| 1024³ | GFLOP/s |
|---|---|
| OpenMat CUDA | 1636 |
| PyTorch CUDA (cuBLAS) | 16869 |

The kernel ([src/ops/kernels/matmul_gpu.cu](src/ops/kernels/matmul_gpu.cu)) is
already tiled — it stages 16×16 tiles of A and B in shared memory with the usual
two `__syncthreads()` per tile. The 10.3× is what is left *after* that, and it
comes from three things the kernel does not do:

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
  throughput lives. `test_benchmarks` prices this directly: fp16 buys only
  ~1.1× over fp32 (1665 vs 1664 GFLOP/s at 1024², 1692 vs 1509 at 2048²),
  where tensor cores would be worth several×.

Global loads are scalar as well — one `T` per thread per tile, strided by
`strideA1` — rather than vectorised; a smaller win, but a free one once the tile
shape allows it.

## 7. CPU elementwise ops were single-threaded — OpenMP closes most of it

`DEFINE_BINARY_OPS_CPU` and `DEFINE_UNARY_OPS_CPU` — the CPU side of
`add`/`sub`/`mul`/`div`, tensor⊕tensor and tensor⊕scalar, in
[headers/ops/cpu/binary_op_macros.h](headers/ops/cpu/binary_op_macros.h) and
[headers/ops/cpu/unary_op_macros.h](headers/ops/cpu/unary_op_macros.h) — were a
single-thread scalar `for` loop, full stop. PyTorch on the same machine uses 20.
Since both macros are a single point of generation, one `#pragma omp parallel for
schedule(static) if(_total > 65536)` in each covers every op they generate at once.

**Methodology note:** this section is measured with
[scripts/bench_omp.py](scripts/bench_omp.py), not `bench_vs.py` — same Release
binary, same code path, `OMP_NUM_THREADS` toggled between invocations as the
single/multi-thread proxy. NumPy is timed in the same process as a reference.

### 7.1 `add`/`sub`/`mul`/`div` — the threshold does its job

`OMP_NUM_THREADS=1` vs 20, same binary:

| op | N | 1 thread | 20 threads | speedup | NumPy | OpenMat vs NumPy |
|---|---|---|---|---|---|---|
| `add` (tensor) | 4096 | 0.0019 ms | 0.0020 ms | 0.95× | 0.0004 ms | 0.20× |
| `add` (tensor) | 65536 | 0.0062 ms | 0.0062 ms | 1.00× | 0.0045 ms | 0.73× |
| `add` (tensor) | 1 048 576 | 0.1111 ms | 0.0209 ms | **5.32×** | 0.1059 ms | **5.07×** |
| `sub` (tensor) | 1 048 576 | 0.1127 ms | 0.0178 ms | **6.33×** | 0.1053 ms | **5.92×** |
| `mul` (tensor) | 1 048 576 | 0.1127 ms | 0.0162 ms | **6.96×** | 0.1059 ms | **6.54×** |
| `div` (tensor) | 1 048 576 | 0.2696 ms | 0.0233 ms | **11.57×** | 0.2670 ms | **11.46×** |
| `add` (scalar) | 1 048 576 | 0.0694 ms | 0.0132 ms | **5.26×** | 0.0573 ms | **4.34×** |
| `mul` (scalar) | 1 048 576 | 0.0693 ms | 0.0104 ms | **6.66×** | 0.0569 ms | **5.47×** |
| `add` (tensor) | 16 777 216 | 4.1551 ms | 1.9323 ms | **2.15×** | 5.6557 ms | **2.93×** |
| `sub` (tensor) | 16 777 216 | 4.1585 ms | 1.9739 ms | **2.11×** | 6.0701 ms | **3.08×** |
| `mul` (tensor) | 16 777 216 | 4.1513 ms | 1.9957 ms | **2.08×** | 5.7388 ms | **2.88×** |
| `div` (tensor) | 16 777 216 | 4.3497 ms | 1.9831 ms | **2.19×** | 5.9544 ms | **3.00×** |
| `add` (scalar) | 16 777 216 | 2.3488 ms | 1.3542 ms | **1.73×** | 3.9418 ms | **2.91×** |
| `mul` (scalar) | 16 777 216 | 2.3269 ms | 1.2635 ms | **1.84×** | 3.9448 ms | **3.12×** |

Below 65536 elements the `if()` clause means fork/join is never paid: every row
there sits within 5 % of the single-thread number, and the 65536 row — at the
threshold, which is `>` not `≥` — is still scalar and identical to 3 decimal
places. Above it the win ranges **1.7–11.6×**, largest at 1 M where the working
set is L2-resident and the scalar loop, not memory, was the ceiling. At 16 M the
buffers exceed cache and the win compresses to ~2×, which is the memory system
talking.

Against NumPy at 20 threads, `add`/`sub`/`mul`/`div` are **2.9–3.1× faster at
16 M and 5.1–11.5× at 1 M**, rather than merely at parity, which is what §3's
allocator fix alone had bought.

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
`min`/`max` still measure identically at 1 and 20 threads — confirming no pragma
of either kind is present:

| op | N | 1 thread | 20 threads | NumPy | OpenMat vs NumPy |
|---|---|---|---|---|---|
| `min` | 1 048 576 | 0.2791 ms | 0.2787 ms | 0.0409 ms | 6.8× slower |
| `max` | 1 048 576 | 0.2796 ms | 0.2793 ms | 0.0409 ms | 6.8× slower |
| `min` | 16 777 216 | 4.7224 ms | 4.7109 ms | 1.7182 ms | 2.7× slower |
| `max` | 16 777 216 | 4.7168 ms | 4.7092 ms | 1.7192 ms | 2.7× slower |

Still 2.7–6.8× slower than NumPy — a real gap, and now the largest remaining one
on the CPU elementwise surface, since §5 closed `sum`. The lever left is a
`parallel for` with a `reduction(min:)`/`(max:)` **across threads** (as opposed
to the `simd` clause within one), which is a different change from the one
reverted above and has not been measured.

### 7.3 `apply`/`apply_binary` are still single-threaded — and still beat NumPy

`relu`, `sigmoid`, `scale_shift`, `shift_scale` and the four `fused_*` methods go
through `Tensor::apply`/`apply_binary` ([headers/tensor.inl](headers/tensor.inl)),
which did not get this pass. Measured at 1 and 20 threads they are identical,
confirming it:

| op | N | 1 thread | 20 threads | NumPy | OpenMat vs NumPy |
|---|---|---|---|---|---|
| `relu` | 16 777 216 | 2.3433 ms | 2.3440 ms | 5.1940 ms | **2.22× faster** |
| `x*s+t` | 16 777 216 | 2.3561 ms | 2.3142 ms | 8.0755 ms | **3.49× faster** |

They beat NumPy anyway, on fusion alone (§1) — which is why this is a missed
opportunity rather than a defect. The same `if(_total > 65536)` treatment would
apply unchanged, and on the §7.1 evidence should be worth roughly 2× at 16 M.

## 8. Elementwise kernels ignored contiguity — every rank now runs at rank-1 speed

§2 raised the rank-1 blocks from 16 threads to 256 and closed the rank-1 gap.
What it did not touch is every *other* rank, and that turned out to be the
larger hole.

The rank-specialized launchers map tensor axes onto grid axes. Only at rank 1
does that leave a warp's 32 lanes contiguous in memory. A rank-2 `dim3(16,16)`
block gives each warp two disjoint runs of 16 elements; a rank-3 `dim3(8,8,8)`
block gives it four runs of 8. Those are 64- and 32-byte requests against a
128-byte line. Rank 5 has no specialized kernel at all and lands on `_nd`, which
reconstructs a multi-index with a `%` and a `/` per axis per element.

The shape was never load-bearing. Every tensor OpenMat produces is contiguous
row-major — `reshape` and friends deep-copy, nothing returns an aliasing view —
so the launchers can throw the axis structure away and index the buffer
linearly. `TensorView::is_contiguous()` gates it, so a strided view (roadmap P2)
falls back to the existing kernels rather than reading the wrong elements.

Kernel time only, 16 M elements, 3× traffic, `add`, GB/s. The "before" rows are
the historical pre-fix measurement, kept for the comparison:

| dtype | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
|---|---|---|---|---|---|
| `float` before | 229.7 | 206.3 | 141.3 | 104.4 | 26.1 |
| `float` after | 229.5 | **228.6** | **232.8** | **229.0** | **231.0** |
| `int` before | 232.9 | 210.3 | 144.4 | | |
| `int` after | 224.2 | **232.4** | **228.0** | | |
| `float16_t` before | 241.8 | 107.5 | 68.0 | | |
| `float16_t` after | 231.5 | **232.1** | **232.2** | | |
| `char` before | 190.8 | 52.7 | 34.0 | | |
| `char` after | **270.1** | **270.2** | **269.8** | | |

`char` reads high across the board because 16 M elements is only 48 MiB of
traffic against a 24 MiB L2 — it is not comparable with the `float` row, only
with the `char` row above it.

`char` is also the one dtype that needed more than flattening. A lane loading
one `char` puts 32 bytes in front of a warp, and that alone costs bandwidth:
190.8 GB/s at rank 1, where the layout was already ideal. So the fast path packs
`4 / sizeof(T)` elements per thread — 1 for `float` and `int`, 2 for
`float16_t`, 4 for `char` — punned through a 4-byte word.

Four bytes, not sixteen. Vector loads are the standard advice and they measured
*worse*: one 16-byte `float4` per thread drops `add` to 216 GB/s against 235 for
one scalar `float`, because the launch gives up the thread-level parallelism it
needs to keep the memory pipeline busy. A grid-stride loop capped at a few waves
per SM costs another 8–12 %. Neither is used. Isolated, 16 M elements:

| bytes per thread | `float` | `float16_t` | `char` |
|---|---|---|---|
| 1 | | | 193 |
| 2 | | 235 | |
| 4 | **235** | **239** | **236** |
| 8 | 227 | 227 | 227 |
| 16 | 216 | 223 | 223 |

Block size stays at 256. 512 and 1024 buy 2–3 % once the working set passes L2
and lose up to 35 % below it, where occupancy rather than bandwidth is the
limit; 256 is never more than 2.5 % off the best at any size.

### Re-measured end to end

[scripts/bench_rank_sweep.py](scripts/bench_rank_sweep.py) reshapes the same
16 M-element buffer and times it through the Python bindings, allocation and
synchronization included, same min-of-7-batches method as the rest of this
report:

| op | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
|---|---|---|---|---|---|
| `add` | 864.8 µs | 861.3 µs | 861.4 µs | 862.3 µs | 862.1 µs |
| `relu` | 579.9 µs | 576.7 µs | 585.3 µs | 577.3 µs | 580.2 µs |
| `fused (a+b)*s` | 864.3 µs | 860.1 µs | 861.6 µs | 861.0 µs | 861.6 µs |

Every op is flat across all five ranks — 229–234 GB/s throughout, a spread of
0.4 % for `add` and 1.5 % for `relu`. The shape no longer influences the time,
which is the whole point. Against
the pre-fix figures the rank-5 `add` improvement is **8.9×** (7644 → 862 µs) and
rank-3 is **1.67×** (1442 → 861 µs).

The knock-on effect is the headline change in this edition: with every rank at
the rank-1 layout, GPU `add` at 16 M now runs at **236.4 GB/s against PyTorch's
232.4** — 1.02× faster, where the previous report had it 1.47× slower. `relu`
went from 1.9× slower to parity, and `x*s+t` from parity to 1.97× faster.

## Full results

### CPU, default threading (20 OpenMP threads)

```
op                             shape      np us  torch us     om us
elementwise/add              (4096,)       0.44      0.77      1.93
elementwise/mul              (4096,)       0.43      0.73      1.92
fused/(a+b)*s                (4096,)       0.96      2.10      1.77
fused/x*s+t                  (4096,)       1.00      2.58      1.82
unary/relu                   (4096,)       1.17      0.68      1.49
reduction/sum                (4096,)       0.61      0.97      0.91
elementwise/add             (65536,)       4.44      9.66      6.25
elementwise/mul             (65536,)       4.48      5.06      6.27
fused/(a+b)*s               (65536,)       7.47      9.08      6.05
fused/x*s+t                 (65536,)       5.45      8.73      5.94
unary/relu                  (65536,)      13.27      6.73      5.59
reduction/sum               (65536,)       6.20      6.33      4.82
elementwise/add           (1048576,)     106.12     16.81     19.51
elementwise/mul           (1048576,)     110.10     15.21     15.95
fused/(a+b)*s             (1048576,)     202.22     55.07    153.82
fused/x*s+t               (1048576,)     199.99     48.63     83.58
unary/relu                (1048576,)     207.58     10.43     82.52
reduction/sum             (1048576,)     100.80     11.37     66.67
elementwise/add          (16777216,)    5637.40   1951.11   1990.04
elementwise/mul          (16777216,)    5681.66   1998.50   1938.47
fused/(a+b)*s            (16777216,)   10488.18   3407.21   4154.81
fused/x*s+t              (16777216,)    8118.93   2831.60   2655.09
unary/relu               (16777216,)    5513.03   1392.53   2298.37
reduction/sum            (16777216,)    1929.27    649.89   1820.77
linalg/matmul             (128, 128)      22.17     15.64     21.34
linalg/matmul             (512, 512)     402.55    417.72   1909.39
linalg/matmul           (1024, 1024)    2795.08   2822.08  17482.03
shape/transpose           (512, 512)     149.19    108.82    181.92
shape/transpose         (2048, 2048)    5286.79   2410.76   7170.53
```

### CPU, `OPENMAT_HOST_CACHE_BYTES=0` (host block cache disabled)

The pre-§3 allocator behaviour, kept as the A/B that shows the block cache is
still load-bearing. This is *not* the default build.

```
op                             shape      np us  torch us     om us
elementwise/add           (1048576,)     103.36     21.85     18.67
elementwise/mul           (1048576,)     104.51     14.73     18.24
fused/(a+b)*s             (1048576,)    1799.86     37.61    108.85
fused/x*s+t               (1048576,)     797.92     24.00     69.17
unary/relu                (1048576,)     204.17     10.14     81.85
reduction/sum             (1048576,)      99.73     10.06     68.44
elementwise/add          (16777216,)    6251.77   1997.86  12270.96
elementwise/mul          (16777216,)    5664.18   1964.04  13071.42
fused/(a+b)*s            (16777216,)   10418.36   3476.06  22430.30
fused/x*s+t              (16777216,)    7899.90   2784.15  22050.22
unary/relu               (16777216,)    5255.17   1354.63  14412.84
reduction/sum            (16777216,)    2096.86    718.54   2119.60
```

### CUDA

```
op                             shape   torch us     om us   om GB/s  torch GB/s
elementwise/add              (4096,)      3.11      7.94       6.2        15.8
elementwise/mul              (4096,)      3.12      7.89       6.2        15.8
fused/(a+b)*s                (4096,)      6.79      8.03       6.1         7.2
fused/x*s+t                  (4096,)      7.03      8.07       4.1         4.7
unary/relu                   (4096,)      3.42      7.68       4.3         9.6
reduction/sum                (4096,)     11.07     12.78       1.3         1.5
elementwise/add             (65536,)      3.17      8.26      95.2       248.4
elementwise/mul             (65536,)      3.18      8.24      95.5       247.7
fused/(a+b)*s               (65536,)      6.92      8.38      93.9       113.6
fused/x*s+t                 (65536,)      7.02      8.34      62.8        74.7
unary/relu                  (65536,)      3.53      7.90      66.4       148.4
reduction/sum               (65536,)     21.42     13.63      19.2        12.2
elementwise/add           (1048576,)      8.20     14.14     889.9      1534.7
elementwise/mul           (1048576,)      8.20     14.12     890.9      1534.6
fused/(a+b)*s             (1048576,)     28.60     14.31     879.0       439.9
fused/x*s+t               (1048576,)     25.58     13.64     615.1       328.0
unary/relu                (1048576,)      8.11     13.13     638.9      1033.9
reduction/sum             (1048576,)     15.35     26.72     157.0       273.2
elementwise/add          (16777216,)    866.45    851.48     236.4       232.4
elementwise/mul          (16777216,)    865.35    851.62     236.4       232.7
fused/(a+b)*s            (16777216,)   1427.77    851.06     236.6       141.0
fused/x*s+t              (16777216,)   1130.38    573.41     234.1       118.7
unary/relu               (16777216,)    569.53    573.20     234.2       235.7
reduction/sum            (16777216,)    272.90    317.03     211.7       245.9
linalg/matmul             (128, 128)      8.80     12.83
linalg/matmul             (512, 512)     28.67    177.44
linalg/matmul           (1024, 1024)    127.30   1312.53
shape/transpose           (512, 512)     10.25     18.21     115.2       204.7
shape/transpose         (2048, 2048)    236.29    204.68     163.9       142.0
transfer/h2d                (65536,)     12.77     13.01      20.1        20.5
transfer/d2h                (65536,)     11.75     11.02      23.8        22.3
transfer/h2d              (1048576,)     79.42     80.83      51.9        52.8
transfer/d2h              (1048576,)     79.37     79.48      52.8        52.8
transfer/h2d             (16777216,)   1131.82   1178.51      56.9        59.3
transfer/d2h             (16777216,)   6814.64  39779.79       1.7         9.8
```

Two notes on reading this table. Sub-1 M GB/s figures are dispatch-bound, not
bandwidth measurements. And the last row is an artifact of end-of-run memory
pressure that hits both libraries — see §3; measured in isolation both do
1.136 ms / 59.1 GB/s.

### In-tree timing suites

`test_benchmarks`, same build:

| MatMul | fp32 ms | fp32 GFLOPS | fp16 ms | fp16 GFLOPS |
|---|---|---|---|---|
| 256² | 0.03 | 1053 | 0.03 | 970 |
| 512² | 0.18 | 1525 | 0.18 | 1530 |
| 1024² | 1.29 | 1664 | 1.29 | 1665 |
| 2048² | 11.38 | 1509 | 10.15 | 1692 |
| 4096² | 104.99 | 1309 | 92.43 | 1487 |

fp16 buys only ~1.1× over fp32, as expected from a hand-written kernel that does
not use the tensor cores (§6).

Element-wise on 16 M elements:

| Op | ms | Gelem/s |
|---|---|---|
| float32 `add` | 3.08 | 5.44 |
| float32 `mul` | 2.96 | 5.67 |
| `scale_shift` | 2.72 | 6.18 |
| `fused_add_mul` | 2.99 | 5.62 |
| float16 `add` | 1.56 | 10.74 |
| float16 `mul` | 1.45 | 11.56 |

`test_stress`, with the previous edition's figures for comparison:

| Scenario | before | now |
|---|---|---|
| 500 × 4 MB alloc+fill+free | 255.0 ms | 192.5 ms |
| 64 tensors live simultaneously (2 MB) | 5.8 ms | 4.7 ms |
| 1000 × (add+mul) on 16 M float | 639.4 ms — 78.7 GB/s | **392.3 ms — 128.3 GB/s** |
| 8 streams × 8 MB mul in parallel | 3.1 ms | 3.0 ms |
| `add` 512 MB + 512 MB | 26.2 ms — 61.5 GB/s | 23.0 ms — 70.1 GB/s |
| 200-deep add chain (4 MB) | 13.2 ms | **3.4 ms** |
| permute rank-6 [4⁶] ×1000 | 8.9 ms | 9.5 ms |
| add+mul 32 MB on CPU | 20.1 ms | **4.9 ms** |
| 100 × 64 MB H2D+D2H round-trips | 3109.1 ms — 4.3 GB/s | **262.7 ms — 51.1 GB/s** |

The last row is the one that matters: the repeated-round-trip transfer number
that both reports flagged as the outstanding bottleneck is closed by the
allocator work (§3), 11.8×.

## Reproducing

```bash
# Release build is mandatory — Debug numbers are meaningless.
export CMAKE_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/$(uname -m)-linux-gnu"
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -C build-release -j$(nproc)

# numpy + torch in a throwaway env; the repo's own package needs neither.
uv venv bench-env
VIRTUAL_ENV=bench-env uv pip install numpy
VIRTUAL_ENV=bench-env uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# the main cross-framework table
OPENMAT_LIB=build-release/OpenMat.so PYTHONPATH=python \
  bench-env/bin/python scripts/bench_vs.py --out bench_results.json

# §3 — same pass with the host block cache off, i.e. the pre-fix allocator
OPENMAT_HOST_CACHE_BYTES=0 \
OPENMAT_LIB=build-release/OpenMat.so PYTHONPATH=python \
  bench-env/bin/python scripts/bench_vs.py --no-cuda --out bench_nocache.json

# §8 — the rank sweep, end to end
OPENMAT_LIB=build-release/OpenMat.so PYTHONPATH=python \
  bench-env/bin/python scripts/bench_rank_sweep.py

# §7 — OpenMP scaling, one invocation per thread count
for t in 1 20; do
  OMP_NUM_THREADS=$t OPENMAT_LIB=build-release/OpenMat.so PYTHONPATH=python \
    bench-env/bin/python scripts/bench_omp.py
done

# in-tree timing suites
./build-release/tests/test_benchmarks
./build-release/tests/test_stress
./build-release/tests/test_stream_perf
```

`--quick` shortens the batches for a smoke run. `bench_results.json` carries the
full environment block plus min/median/reps per case.
