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

Timing lives in three GoogleTest suites, not in a separate benchmark script:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..   # Debug numbers are meaningless
make -j$(nproc)
./tests/test_benchmarks     # matmul (fp32/fp16), elementwise, reductions
./tests/test_stream_perf    # the stream experiments tabulated below
./tests/test_stress         # soak / leak checking
```

There is no cross-framework comparison against NumPy or PyTorch yet — see the
[roadmap](#roadmap). The measured stream results below are the numbers this project
actually stands behind; raw output is in [stream_perf_report.md](stream_perf_report.md).

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

Benchmarked on **NVIDIA GeForce RTX 4060 · CUDA 11.5** (`tests/test_stream_perf.cpp`):

#### Single op, sync after each iteration — no improvement

| Variant | Time/iter | Speedup |
|---|---|---|
| `operator+` (sync) | 0.42 ms | 1.00× |
| `add(default_stream())` | 0.42 ms | 1.00× |
| `add(Stream s)` + sync | 0.40 ms | ~1.05× |

When the host blocks after every single operation the round-trip cost is identical regardless of whether a stream is used. The stream wrapper adds no measurable overhead, but it also cannot help if you synchronize immediately.

#### Sequential chain of dependent ops — **2.68× faster**

| Variant | Total time (100 adds, 8 MB) | Speedup |
|---|---|---|
| Sync after every op | 45.38 ms | 1.00× |
| One stream, one sync at the end | 16.94 ms | **2.68×** |

This is the highest-impact use case. With 100 explicit synchronizations the host stalls 100 times; each stall costs ~0.28 ms of scheduling overhead. Enqueuing all 100 kernels on a single stream and syncing once eliminates 99 of those stalls. The GPU's internal ordering guarantees correctness even with data dependencies between ops.

The pattern in practice:

```cpp
om::Stream s;
Tensor<float> x = input;
for (int i = 0; i < 100; ++i)
    x = x.add(bias, s);   // all 100 kernels enqueued without blocking
s.synchronize();           // one round-trip at the end
```

#### Parallel fan-out of independent ops — no improvement on memory-bound ops

| K | Sequential | K parallel streams | Speedup |
|---|---|---|---|
| 2 | 0.15 ms | 0.14 ms | ~1.05× |
| 8 | 0.73 ms | 0.69 ms | ~1.06× |
| 16 | 1.46 ms | 1.38 ms | ~1.06× |

Launching K independent kernels on K streams does not produce meaningful speedup when each kernel already saturates the available memory bandwidth. The RTX 4060 has one compute pipeline; `mul` on 4 MB data is entirely memory-bound and consumes close to 100% of bandwidth by itself. Parallel streams help more with compute-bound or small kernels, or on server GPUs with multiple independent SM partitions.

#### Compute + transfer overlap — **1.12× faster**

| Variant | Total time (20 rounds, 16 MB) | Speedup |
|---|---|---|
| Serialized (H2D → sync → compute → sync) | 37.21 ms | 1.00× |
| Overlapped (stream_copy ∥ stream_compute) | 33.11 ms | **1.12×** |

The RTX 4060 has a dedicated DMA copy engine that operates independently from the compute SMs. Assigning H2D transfers to one stream and compute work to another lets both run simultaneously:

```
Serialized:  [H2D -------][compute -------][H2D -------][compute -------]
Overlapped:  [H2D -------]
                  [compute -------]
                               [H2D -------]
                                    [compute -------]
```

The theoretical maximum speedup is ~2× (total time drops to `max(H2D, compute)` instead of `H2D + compute`). The observed 12% is lower because H2D and compute are similar in duration on this workload, leaving limited asymmetry to exploit. In inference pipelines that alternate between data loading and computation the gain is more pronounced.

#### Stream creation overhead — negligible

| Variant | Time/iter (256 KB mul, 1000 iters) |
|---|---|
| Reuse one stream | 0.01 ms |
| New stream per call | 0.01 ms |

At this scale `cudaStreamCreate` overhead is within measurement noise. For latency-sensitive tight loops (kernel time < 10 µs) the recommendation is still to create streams once and reuse them, as driver overhead becomes a larger fraction of total time.

### Summary

| Pattern | Use streams? | Effect |
|---|---|---|
| Single op, immediate sync | Irrelevant | No change |
| Chain of N ops on the same data | **Yes — one stream, one sync** | Up to ~2.7× faster |
| Independent ops on separate data | Only if compute-bound | ~1× on memory-bound workloads |
| Compute while transferring data | **Yes — two streams** | 10–15% on consumer GPU |
| Stream creation per call | No — reuse streams | No overhead at scale |

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

**RAII for GPU memory via `Tensor<T>` + `Allocator<T>`**
Rather than pairing raw `cudaMalloc`/`cudaFree` calls at each use site, every `Tensor` owns a polymorphic `Allocator` chosen at construction time by `AllocatorFactory`. The destructor delegates to `allocator->deallocate`, making GPU memory lifetime deterministic regardless of exceptions or early returns — the same pattern used in PyTorch's `at::DataPtr`.

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
`CMAKE_CUDA_ARCHITECTURES=native`), a C++17/CUDA 17 compiler. Verified on CUDA 13.0 /
GCC 13.3 / CMake 3.28 / GB10 (sm_121), all 12 suites passing.

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
cd build && ctest                     # all 12 suites
./tests/test_arithmetic               # a single suite, per-test output
./tests/test_arithmetic --gtest_filter="TensorArithmetic.CPUOperations"
```

`test_arithmetic`, `test_fused_ops`, `test_device_transfer`, `test_factory`,
`test_reductions`, `test_reshape`, `test_transpose`, `test_streams`, `test_allocator_stream`
are correctness suites. `test_benchmarks`, `test_stress` and `test_stream_perf` are
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
- [ ] Benchmark suite comparing against NumPy / PyTorch
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
- **Streams only help when you stop synchronizing** — the single biggest win (2.68×) comes not from running kernels faster but from removing 99 out of 100 host/device sync barriers in a chain. The GPU did the same work; the host just stopped blocking between submissions.
- **Raw arrays decay in CUDA kernel parameters** — passing `size_t axes[MAX_RANK]` as a kernel argument silently becomes a host pointer on the device side. The fix is a trivially-copyable struct wrapper so CUDA copies the data by value into the kernel parameter block.
- **A Python reference is not a lifetime guarantee** — the first two attempts at keeping a CUDA stream alive from Python both segfaulted under `gc.collect()`. The cyclic collector finalizes a cycle's members, plus everything reachable only from them, in arbitrary order, so a `Stream` could be torn down while tensors still owed it a stream-ordered free. Reference counting had to move to the C side of the boundary.
- **`cudaMallocAsync` is not a drop-in replacement** — it uses a stream-ordered memory pool. Freeing on a different stream than the one used for allocation is a programming error that manifests as an illegal memory access with no obvious call site. Storing `m_Stream` in each `Tensor` and using it in the destructor is the invariant that keeps this safe.

---

## License

MIT