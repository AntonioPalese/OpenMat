# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Requirements: NVIDIA GPU, CUDA Toolkit ≥ 11.2 (for `cudaMallocAsync`), CMake ≥ 3.24 (for `CMAKE_CUDA_ARCHITECTURES=native`), C++17/CUDA 17 compiler, OpenMP (`find_package(OpenMP REQUIRED)` in `CMakeLists.txt`; bundled as `libgomp` with a stock GCC install, nothing extra to install on most systems). Verified building and passing all 14 suites on CUDA 13.0 / GCC 13.3 / CMake 3.28 / GB10 (sm_121) — the 11 correctness suites run in 4.7 s; the three timing/soak suites are separate.

```bash
# Full clean rebuild (also refreshes compile_commands.json in the repo root)
./compile.sh

# Or manually:
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make -j$(nproc)
```

Produces `build/OpenMat.so` (shared library — also what the Python package loads), `build/OpenMat_app`, and `build/tests/test_*`.

Build notes that bite:
- **`CMAKE_LIBRARY_PATH` must be set in the environment** (colon-separated); its entries become `target_link_directories` for `-lcuda`/`-lcudart`. CMake only warns when it is unset — `OpenMat.so` still builds, then `OpenMat_app` and every test binary fail with `cannot find -lcudart`. Typically:
  ```bash
  export CMAKE_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/$(uname -m)-linux-gnu"
  ```
- `CMAKE_CUDA_ARCHITECTURES` defaults to `native` (the GPU in this machine); override with `-DCMAKE_CUDA_ARCHITECTURES=<sm>` to cross-compile. The guard sits **above `project()`** on purpose: `project(... LANGUAGES CUDA)` defines the variable with CMake's own default, so an `if(NOT DEFINED ...)` placed after it silently never fires — you get the default arch (75 on CUDA 13) and PTX-JIT on every process start instead of native SASS.
- [cmake/detect_cuda_arch.cmake](cmake/detect_cuda_arch.cmake) and [scripts/detect_archs.py](scripts/detect_archs.py) are stale and unused — the former shells out to `nvcc --list-gpus`, an option removed in CUDA 13, and would parse the lowest *supported* arch rather than the local GPU's anyway. `native` replaces both.
- Default `CMAKE_BUILD_TYPE` is `Debug`. Pass `-DCMAKE_BUILD_TYPE=Release` before benchmarking — the numbers in [README.md](README.md) and [stream_perf_report.md](stream_perf_report.md) are Release numbers.
- Sources are picked up with `file(GLOB ...)`; a newly added `.cpp`/`.cu` needs a CMake re-run, not just `make`.

## Tests

GoogleTest, fetched by CMake via FetchContent. One binary per suite, registered in [tests/CMakeLists.txt](tests/CMakeLists.txt) via `add_om_test`.

```bash
cd build && ctest                       # all suites
./build/tests/test_arithmetic           # one suite, per-test output
./build/tests/test_arithmetic --gtest_filter="TensorArithmetic.CPUOperations"
```

Suites: `test_arithmetic`, `test_fused_ops`, `test_device_transfer`, `test_factory`, `test_reductions`, `test_benchmarks`, `test_reshape`, `test_transpose`, `test_streams`, `test_allocator_stream`, `test_host_pool`, `test_contiguous`, `test_stress`, `test_stream_perf`.

`test_benchmarks`, `test_stress`, and `test_stream_perf` are timing/soak suites, not correctness suites — they are slow and their numbers are meaningless in a Debug build. `StreamPerf.ParallelFanOut` in particular asserts wall-clock against wall-clock and goes red under load; CI does not gate on those two suites.

**Every test that touches the device starts with `OM_REQUIRE_CUDA();`** ([tests/test_helpers.h](tests/test_helpers.h)) — a `cudaGetDeviceCount` check that `GTEST_SKIP`s instead of failing. That is what makes `ctest` green on a machine with no GPU (110 skipped, 60 host-side tests run) and is what the CPU-only CI job relies on; a new GPU test without it turns that job red. The Python equivalents are the `requires_cuda` marker and the `device` fixture in [python/tests/conftest.py](python/tests/conftest.py).

## Debugging asynchronous kernel errors

`OPENMAT_DEBUG_SYNC=1` forces a `cudaStreamSynchronize` after every kernel launch, so an out-of-bounds access is reported at the launching call site with the kernel's name instead of surfacing later as an illegal access in an unrelated call:

```bash
OPENMAT_DEBUG_SYNC=1 ./build/tests/test_streams          # no rebuild needed
```

```
[CUDA ASYNC ERROR] kernel 'add_kernel_rank1' at src/ops/kernels/binary_ops.cu:10
  in void om::launch_add(...) [with T = float; ...]
  on stream 0xb0149ef4a290
  → an illegal memory access was encountered
```

`cmake -DOM_DEBUG_SYNC=ON` makes it the build's default (the env var still overrides either way, so `OPENMAT_DEBUG_SYNC=0` turns it back off); `cmake -DOM_NO_DEBUG_SYNC=ON` compiles it out entirely. It serializes streams — diagnostic mode, never a default. It complements `compute-sanitizer`: cheap enough to leave on for a whole test run, but it only localizes, it does not tell you which access was bad.

**Every kernel launch must be followed by `CUDA_CHECK_LAUNCH(kernel_name, stream)`** ([headers/cuda_defines.cuh](headers/cuda_defines.cuh)), not the older bare `CUDA_CHECK` — that one has no way to know the stream or the kernel, so it can only do the synchronous check. In the rank-switching launcher macros the name is carried in a local `const char* om_kernel` assigned in each branch, with one check after the `switch`. The implementation is deliberately non-`inline`, in [src/cuda_debug.cpp](src/cuda_debug.cpp): both switches are preprocessor-conditional, and an inline definition would let a consumer built with different settings supply a conflicting second definition (ODR violation, linker silently picks one).

## CI

[.github/workflows/ci.yml](.github/workflows/ci.yml), on every push and PR. Two jobs:

- **cpu** — a `nvidia/cuda:*-devel` container on a GitHub-hosted runner: toolkit, no driver, no device. It configures with an explicit `-DCMAKE_CUDA_ARCHITECTURES="75;86;89"` (the `native` default needs a GPU to resolve) and puts `/usr/local/cuda/lib64/stubs` first in `CMAKE_LIBRARY_PATH`, where the link-time-only `libcuda.so` lives. Its job is the full compile+link — the macro machinery and the mandatory explicit instantiations only fail there — plus the host-side tests.
- **gpu** — a self-hosted runner labelled `self-hosted,linux,gpu`, needing nvcc and CMake ≥ 3.24 on PATH. Release build at `native` arch, the correctness suites (`ctest -E "test_benchmarks|test_stream_perf"` — 12 of the 14, `test_stress` included), the Python suite, then `compute-sanitizer --tool memcheck --leak-check full` over `test_stress`, `test_allocator_stream` and `test_streams` (~30 s total). memcheck is the only thing that reports a stream-ownership violation near the call site instead of as an illegal access in an unrelated kernel later on.

## Python package

The package is a **ctypes** binding (not pybind) over the C-ABI layer compiled into `OpenMat.so`.

```bash
cd python
uv sync
uv pip install -e .
```

[python/openmat/_clib.py](python/openmat/_clib.py) locates the library in this order: `$OPENMAT_LIB` → `openmat/OpenMat.so` bundled in the wheel → `<repo>/build/OpenMat.so`. The third fallback means the bindings work straight from a source checkout after `./compile.sh` with no install step:

```bash
OPENMAT_LIB=build/OpenMat.so python python/test_bindings.py   # smoke script
cd python && pytest                                            # pytest suite (python/tests/)
```

Python suites: `test_tensor.py` (the original surface), `test_tensor_api.py` (metadata, indexing, shape/fused ops, buffer protocols), `test_dtypes.py`, `test_streams.py`. [python/tests/conftest.py](python/tests/conftest.py) provides a `device` fixture that runs a test on both backends and a `requires_cuda` marker.

[python/hatch_build.py](python/hatch_build.py) is the custom hatch hook `pyproject.toml` points at: it copies `build/OpenMat.so` (or `$OPENMAT_LIB`) into `openmat/` so the wheel bundles it, and only warns if no library is present — an sdist should not need a CUDA toolchain. The copied `openmat/OpenMat.so` is gitignored by the root `*.so` rule.

Gotcha: the root [.gitignore](.gitignore) starts with `*build*`, which matches `hatch_build.py` itself — that is why the file was absent from the repo and `uv pip install -e .` failed. It is now kept by an explicit `!python/hatch_build.py`; be careful adding any other source file with "build" in its name.

## Architecture

Everything lives under the `om` namespace.

### Streams are the canonical execution path

Every `Tensor<T>` operation exists in two forms and the stream form is the real implementation:

```cpp
auto c = a + b;                    // delegates to a.add(b, Stream::default_stream())
auto c = a.add(b, s);              // enqueues on s; caller must s.synchronize()
```

`om::Stream` ([headers/stream.h](headers/stream.h)) is a move-only RAII wrapper. The default constructor calls `cudaStreamCreate` and owns the handle; `Stream(cudaStream_t)` wraps an existing handle without owning it; `Stream::default_stream()` returns a non-owning wrapper around `nullptr`, which is how the synchronous API reuses the async code path with zero duplication.

**When adding an op, implement the `(args, const Stream&)` overload and make the no-stream one a one-line delegate.** Doing it the other way round breaks the single-source-of-truth invariant the whole `tensor.inl` is built on.

### Dispatch: two paths, one of them mostly dead

[headers/kernel_launcher.h](headers/kernel_launcher.h)/[.inl](headers/kernel_launcher.inl) still define the macro-generated dispatch machinery:
- `DEFINE_DEVICE_DISPATCH_BINARY_H(OP, CPU_FUNC, CUDA_FUNC)` — declares `OP_dispatch<DEVICE_TYPE, T>` structs.
- `DEFINE_DEVICE_DISPATCH_BINARY_INL(OP)` — defines the free function `_OP(lhs, rhs, dst, DEVICE_TYPE)` that switches at runtime.
- `DEFINE_DEVICE_DISPATCH_UNARY_H/INL` — same for tensor⊕scalar ops.

**But `Tensor<T>`'s stream overloads no longer go through them.** Since the stream refactor, [headers/tensor.inl](headers/tensor.inl) branches on `device_type()` itself and calls `add_cpu(...)` / `launch_add(..., s.get())` directly, because the `_dispatch` structs have no `cudaStream_t` parameter. Today only `_fill` still uses the dispatch path (from `Tensor::fill`). Treat the `_dispatch` macros as legacy: adding a new op means wiring `tensor.inl` directly, and adding a dispatch registration only if you also need the stream-less free function.

Effective data flow for `a + b`:

```
Tensor<T>::operator+()
  → Tensor<T>::add(rhs, Stream::default_stream())          [tensor.inl]
    → add_cpu(...)  or  launch_add(..., stream)            [ops/cpu/ or ops/kernels/]
      → flat CPU loop  or  rank-specialized CUDA kernel
```

### Memory and views

**`Tensor<T>`** ([headers/tensor.cuh](headers/tensor.cuh), [headers/tensor.inl](headers/tensor.inl)) — owning N-D tensor: shape, row-major strides, raw `T*`, a `Device`, an `om::Stream m_Stream`, and a `unique_ptr<Allocator<T>>`. Copy deep-copies; move transfers ownership and nulls the source. There is a private `Tensor(shape, device, Stream)` constructor used by the stream overloads so a result tensor is allocated and freed on the stream that produced it.

**`Allocator<T>` / `AllocatorFactory<T>`** ([headers/allocator.h](headers/allocator.h), [.inl](headers/allocator.inl)) — `CpuAllocator` (host block cache/memcpy) and `GpuAllocator` (cudaMalloc/cudaFree/cudaMemcpy). The base class declares `allocate_async`, `deallocate_async`, `copy_async`, `copy_host_to_device_async`, `copy_device_to_host_async` with **synchronous default implementations**, so a subclass overrides only what it can actually do async. `GpuAllocator` uses `cudaMallocAsync`/`cudaFreeAsync` under `#if CUDART_VERSION >= 11020`.

**Host memory is recycled, not returned to the OS.** `CpuAllocator::allocate/deallocate` go through `om::detail::HostPool` ([headers/host_pool.h](headers/host_pool.h)), a process-wide free list keyed by size class. Plain `malloc` was measurably the dominant cost of every host-side op: above glibc's 128 KB `MMAP_THRESHOLD` each allocation is a fresh `mmap`, so an out-of-place op page-faults its whole output buffer before computing anything — 16384 faults per 64 MB result. Recycling a block keeps its pages mapped and took `add` at 16 M elements from 19.4 ms to 4.5 ms and a 64 MB `Tensor::cpu()` from 18.2 ms to 1.14 ms (the D2H case pays the faults *inside* `cudaMemcpy`, which is why the allocator showed up as a transfer problem).

Details that matter if you touch it: requests round up to a size class (8 per octave, ≤ 12.5 % overshoot) so the number of free lists stays bounded; each block carries a 64-byte header holding its class, so `deallocate` needs no side table and the pointer handed out is 64-byte aligned rather than malloc's 16; the cache is capped (256 MB by default, `OPENMAT_HOST_CACHE_BYTES` overrides, `0` disables recycling and restores the old behaviour — useful for A/B measurement); and the singleton is deliberately never destroyed, because a `Tensor` with static storage duration would otherwise free into a destroyed pool. Host pointers must therefore never be freed with bare `std::free`, and host memory must never be allocated with bare `malloc` and handed to a `Tensor`.

**Pinned (page-locked) host memory is opt-in, not automatic.** `PinnedCpuAllocator<T>` (subclasses `CpuAllocator<T>`, overrides only `allocate`/`deallocate`) allocates through `om::detail::PinnedHostPool` — the same size-class recycling as `HostPool`, but `cudaHostAlloc`/`cudaFreeHost` instead of `malloc`/`free`, capped separately (`OPENMAT_PINNED_CACHE_BYTES`, default 64 MB) because page-locking is one to two orders of magnitude slower than a pageable allocation, so recycling matters even more here. Nothing decides on its own which host tensors will cross the bus — a `Tensor` gets `PinnedCpuAllocator` only via `Tensor::pinned(shape)` (an explicit request, for a buffer known to be a repeated H2D *source*) or as the destination `Tensor::to()` allocates for a device-to-host copy, where it isn't a guess: that exact buffer's only purpose is to receive that exact copy. `device_type()` still reports `CPU` either way — there is no third `DEVICE_TYPE`; pinned-ness is purely which allocator subclass `m_Allocator` holds, queryable via `Tensor::is_pinned()` (a `dynamic_cast`). Both call sites already require a working CUDA driver, so unlike `HostPool` this pool is never touched by the CPU-only CI job — it still has to compile there (against the stub `libcuda.so`), but nothing exercises it. One consequence of using a CUDA-tracked allocation API for a "never destroyed" singleton: `compute-sanitizer --leak-check full` (which the gpu CI job runs with `--error-exitcode 1`) would flag every still-cached block as leaked at exit, unlike `HostPool`'s plain-`malloc` cache, which the sanitizer can't see. `PinnedHostPool`'s constructor registers an `atexit` hook that empties the free list on the way out; it only calls `release_all()`, never destroys the pool object, so it doesn't reopen the destruction-order hazard the singleton pattern exists to avoid. See [benchmark_report.md §3](benchmark_report.md#3-the-cpu-gap-above-128-kb-was-the-allocator-not-the-loop--fixed) for what this does and does not move on the reference hardware.

**Stream-ownership invariant:** `cudaMallocAsync` memory belongs to a stream-ordered pool and must be freed on the stream it was allocated on. That is why each `Tensor` stores `m_Stream` and the destructor calls `deallocate_async(m_Data, m_Stream.get())`. Breaking this shows up as an illegal memory access far from the real call site.

**`TensorView<T>`** ([headers/tensor_view.cuh](headers/tensor_view.cuh)) — non-owning host-side view (pointer + shape/stride pointers + rank), `__host__`-only. Converted with `.as_device_tw()` at launch.

**`DeviceTensorView<T>`** ([headers/device_tensor_view.cuh](headers/device_tensor_view.cuh)) — non-owning device-side view. Shape and stride are **fixed inline arrays** (`size_t shape[MAX_RANK]`, `MAX_RANK = 8`) filled from host at construction, so the struct is trivially copyable and passed **by value** into the kernel parameter block. There is deliberately no device allocation here — an earlier design cudaMalloc'd shape/stride per view (6 allocations per binary op). Do not reintroduce pointer members: raw arrays passed as kernel arguments decay to host pointers on the device side. Rank > 8 trips the constructor `assert`.

**`Device`** ([headers/mat_utils.h](headers/mat_utils.h)) — `m_Id`, `m_Str`, `m_Dt`. Constructible from `(id, DEVICE_TYPE)` or a string like `"cuda:0"`.

### Rank-specialized CUDA kernels

[headers/ops/kernels/binary_op_macros.cuh](headers/ops/kernels/binary_op_macros.cuh): `DEFINE_BINARY_OP_LAUNCH(OP)` generates `launch_OP(lhs, rhs, dst, cudaStream_t)` which switches on `lhs.rank` and picks a kernel with a rank-tuned grid/block layout (1D `dim3(16)`, 2D `dim3(16,16)`, 3D/4D `dim3(8,8,8)`). Rank ≥ 5 falls back to `OP_kernel_nd`, a flat 1D kernel reconstructing multi-indices from a linear index. `DEFINE_BINARY_OP_LAUNCH_FRW_DEC(OP)` emits explicit instantiations for `float`, `int`, `char`, `float16_t`.

**The `_nd` kernel is also the overflow path, not just the rank ≥ 5 path.** `gridDim.y` and `gridDim.z` are capped at 65535 (only `gridDim.x` reaches 2^31-1), so a leading axis large enough overflows the rank-specialized layout — the rank-4 launcher sets `blocks.z = shape[0]`, the rank-3 one `(shape[0] + 7) / 8`. Past that the launch fails synchronously with `invalid configuration argument`, which surfaces to the caller as a generic CUDA error naming nothing useful. Every rank-specialized launcher therefore computes its grid extents as `size_t` and gates the launch on `om::detail::grid_fits(gx, gy, gz)` ([headers/cuda_defines.cuh](headers/cuda_defines.cuh)), falling through to the flat `_nd` kernel when they do not fit. The extents are checked *before* being narrowed into a `dim3` — its members are `unsigned int`, so building one first could truncate an oversized extent into a plausible small one. The same guard is in the unary launcher, `launch_fill`, `launch_apply_op` and `launch_apply_binary_op`; a new rank-switching launcher needs it too.

**Every elementwise launcher tries a contiguous fast path first.** [headers/ops/kernels/contiguous.cuh](headers/ops/kernels/contiguous.cuh) — the rank-specialized layouts only keep a warp's 32 lanes contiguous in memory at rank 1. A rank-2 `dim3(16,16)` block gives each warp two disjoint runs of 16 elements, a rank-3 `dim3(8,8,8)` block four runs of 8: 64- and 32-byte requests against a 128-byte line. Measured on the reference GB10, `add` over 16 M floats ran at 230 GB/s at rank 1, 206 at rank 2, 141 at rank 3, 104 at rank 4 and 26 at rank 5 (`_nd`, which also recomputes a multi-index per element) — same traffic, same op. Since every tensor the library builds is contiguous row-major (reshape and friends deep-copy, nothing returns an aliasing view), the shape carries nothing the kernel needs, so the fast path indexes the buffer linearly and gives every rank the rank-1 layout: all of them now measure 228-233 GB/s. `launch_add`/`launch_sub`/`launch_mul`/`launch_div`, the `_k` scalar family, `launch_apply_op`, `launch_apply_binary_op` and `launch_fill` all take it.

`TensorView::is_contiguous()` is what gates it — the launchers ask rather than assume, so a strided view (roadmap P2) falls back to the existing rank-specialized kernels instead of silently reading the wrong elements. That is the whole reason the guard is a runtime check and not a comment.

Two things about it are counter-intuitive and were measured, not assumed:

- **The pack width is 4 bytes per thread, not 16.** `float4` vector loads are the standard advice and they are *slower* here — one 16-byte `float4` per thread drops `add` to 216 GB/s against 235 for one scalar `float`, because the launch loses the thread-level parallelism that keeps the memory pipeline full. A grid-stride loop capped at a few waves per SM costs a further 8-12 %; at the exact block count its loop body runs once and buys nothing over a bounds check. Neither is used. What does matter is that no thread moves *less* than 4 bytes: a warp of `char` lanes requests 32 bytes and reaches only 193 GB/s even at rank 1. Hence `pack_width<T> = 4 / sizeof(T)` — 1 for `float`/`int`, 2 for `float16_t`, 4 for `char`, which takes `char` to 236 GB/s. The pack is punned through a `unsigned int` with `memcpy`, not a union or a member-wise struct copy: `float16_t` has a user-provided constructor, so a union of it is ill-formed and a member-wise copy is free to lower to two 2-byte accesses, which is exactly what the pack exists to avoid.
- **A size that is not a multiple of the pack width leaves a tail**, picked up by block 0 after the packed loop. `test_contiguous` runs every dtype at sizes covering all four residues mod 4 precisely because nothing else in the suite would notice the tail being dropped.

Block size is 256, as elsewhere in the library. 512 and 1024 buy 2-3 % once the working set passes L2 and lose up to 35 % below it, where occupancy rather than bandwidth is the limit. Re-measure before changing it.

The kernel definitions and the launch statements sit behind `#if defined(__CUDACC__)`: `tensor.cuh` pulls this header into plain `.cpp` translation units (the Python C-ABI layer among them), where `__global__` expands to nothing and `blockIdx` does not exist. A new elementwise launcher that wants the fast path calls `om::detail::launch_contiguous_binary/unary/fill`, which return the launched kernel's name for `CUDA_CHECK_LAUNCH` or `nullptr` when they decline — declining is not an error, it is how an empty tensor, an unaligned buffer or an unrepresentable grid keeps the old, always-correct path. The four generated op families need one more piece: `DEFINE_BINARY_OP_FUNCTOR_H` / `DEFINE_UNARY_OP_FUNCTOR_H` turn the op's expression into a functor type, because the rank-specialized kernels take it textually but the fast path is generic over the operation. See [benchmark_report.md §8](benchmark_report.md#8-elementwise-kernels-ignored-contiguity--every-rank-now-runs-at-rank-1-speed).

**Ops layout:**
```
headers/ops/cpu/        ← CPU op declarations (macro-generated inline functions)
src/ops/cpu/            ← CPU op .cpp translation units
headers/ops/kernels/    ← CUDA kernel declarations and launch macros (.cuh)
src/ops/kernels/        ← CUDA kernel .cu translation units
```

**Adding a new binary op:**
1. `src/ops/kernels/binary_ops.cu` — kernel bodies via `DEFINE_BINARY_OP_KERNEL_K1/K2/K3/K4/ND` + `DEFINE_BINARY_OP_LAUNCH` + `DEFINE_BINARY_OP_LAUNCH_FRW_DEC`.
2. `headers/ops/kernels/binary_op_macros.cuh` — `DEFINE_BINARY_OP_LAUNCH_H` / `DEFINE_BINARY_OP_KERNEL_H`, plus `DEFINE_BINARY_OP_FUNCTOR_H(OP, expr in a and b)` — the launch macro references `OP_fn<T>` for the contiguous fast path, so omitting it is a compile error, not a silent slow path.
3. `src/ops/cpu/binary_ops.cpp` + `headers/ops/cpu/binary_op_macros.h` — CPU side.
4. `headers/tensor.cuh` / `.inl` — the `(rhs, const Stream&)` method plus the one-line no-stream delegate.
5. Only if a stream-less free function is wanted: register in `kernel_launcher.h`/`.inl`.

**CPU binary/unary elementwise ops parallelize above a size threshold.** `DEFINE_BINARY_OPS_CPU` ([headers/ops/cpu/binary_op_macros.h](headers/ops/cpu/binary_op_macros.h)) and `DEFINE_UNARY_OPS_CPU` ([headers/ops/cpu/unary_op_macros.h](headers/ops/cpu/unary_op_macros.h)) — the CPU side of `add`/`sub`/`mul`/`div`, both tensor⊕tensor and tensor⊕scalar — wrap their loop in `#pragma omp parallel for schedule(static) if(_total > 65536)`. Since every op generated from these two macros shares the one loop, all of them benefit together. `_Pragma` is used instead of a bare `#pragma` because the pragma sits inside a macro replacement list — `#pragma` cannot appear mid-macro, `_Pragma` can, and it is expanded at the same point. Below the threshold the loop is measurably untouched (within ~2% of the single-thread time — no fork/join tax paid on the hot path for small tensors); above it, `add`/`sub`/`mul`/`div` measured 1.7–11.6× faster on a 20-thread reference machine (largest at 1 M, where the working set is L2-resident and the scalar loop rather than memory was the ceiling; ~2× at 16 M, which is the memory system talking) and beat NumPy outright by 2.9–3.1× at 16 M. See [benchmark_report.md §7](benchmark_report.md#7-cpu-elementwise-ops-were-single-threaded--openmp-closes-most-of-it) for the numbers. `matmul_cpu` ([headers/ops/cpu/matmul_cpu.h](headers/ops/cpu/matmul_cpu.h)) is parallelized the same way (`#pragma omp parallel for` over the outer row loop, on top of `ikj`-order/L2-tiled inner loops) and predates this.

Every consumer that includes `tensor.cuh` — and therefore re-instantiates these header-only templates — must compile with `-fopenmp` for exactly this reason: `CMakeLists.txt` does `find_package(OpenMP REQUIRED)` and links the `OpenMat` target `PUBLIC` against `OpenMP::OpenMP_CXX`, so the flag propagates to every TU that consumes the headers (tests, `src/main.cpp`, the Python capi TU). Compiling one instantiation with `-fopenmp` and another without is an ODR violation on the same weak symbol, not just a missed optimization.

**Division goes through one policy.** [headers/ops/div_policy.h](headers/ops/div_policy.h) defines `om::div_elem`, used by `div`, `div_k` and the `Div`/`BinaryDiv` fused functors on both backends. Floating point divides unguarded — IEEE 754 already gives ±inf with the dividend's sign and NaN for 0/0, which is what NumPy and PyTorch return; integer types return 0 for `x / 0` (UB otherwise, and NumPy's answer). Do not reintroduce a `rhs != 0 ? … : INFINITY` guard: it loses the sign, disagrees between CPU and GPU for `int`, and the `static_cast<double>` it needs lands on the 1:64 fp64 unit.

**Supported dtypes** (`om::dtype<T>()`): `float`, `double`, `int`, `char`, `float16_t`. Kernel instantiations cover `float`, `int`, `char`, `float16_t` — `double` has no GPU instantiation.

`float16_t` ([headers/type_traits/types.cuh](headers/type_traits/types.cuh)) is a hand-rolled `__half` wrapper, not a CUDA type: it carries `__host__ __device__` conversions plus free `+ - * /` operators that use `__hadd`/`__hsub`/... on `__CUDA_ARCH__ >= 530` and fall back to `float` math otherwise. Generic code gates on `is_extended_arithmetic<T>` (same file) — `std::is_arithmetic` plus a `float16_t` specialization — so a `static_assert` on `std::is_arithmetic` alone will reject half precision.

## Fused operations

[headers/ops/kernels/fused_op.cuh](headers/ops/kernels/fused_op.cuh) — functor-based fusion, no intermediate allocation:

- `Add<T>`, `Mul<T>`, `Div<T>`, `Pow<T>`, `ReLU<T>`, `Sigmoid<T>` — unary functors
- `Compose<F,G>` — `g(f(x))`; uses an explicit `decltype` return type (C++17, not `auto` parameters)
- `BinaryAdd/Sub/Mul/Div<T>`, `BinaryCompose<BinOp,UnaryOp>` — binary functors and binary-then-unary chains
- `launch_apply_op<T>(src, dst, op, stream)` / `launch_apply_binary_op<T>(lhs, rhs, dst, op, stream)` — rank 1–4 kernels plus an `_nd` fallback

**Explicit instantiations** in [src/ops/kernels/fused_op.cu](src/ops/kernels/fused_op.cu) must list every `(T, Op)` pair used from a `.cpp` translation unit. A new functor or a new `Compose` combination without an instantiation is a link error. Calls from `.cu` files instantiate implicitly and hide the problem.

`Tensor<T>` surface: `apply(op[, stream])`, `apply_binary(rhs, op)`, `scale_shift`, `shift_scale`, `relu`, `sigmoid`, `fused_add_mul`, `fused_sub_mul`, `fused_mul_add`, `fused_div_add`. `apply` and `apply_binary` both have a real CPU loop branch — the CUDA-only limitation noted in [docs/roadmap.md](docs/roadmap.md) §4.2 has been fixed.

## Reductions

GPU: two-phase shared-memory tree reduction + warp shuffle (`__shfl_down_sync`) — `launch_reduce_sum/min/max` in [headers/ops/kernels/reduce_gpu.cuh](headers/ops/kernels/reduce_gpu.cuh). CPU: [headers/ops/cpu/reduce_cpu.h](headers/ops/cpu/reduce_cpu.h). Exposed as `.sum()`, `.mean()`, `.min()`, `.max()`; these are synchronous and return a host scalar (no stream overloads).

`reduce_sum_cpu` splits the accumulation across 8 independent lanes (a source-level restructuring, commented inline) to break the loop-carried FP dependency chain — a single accumulator runs at one add per FP *latency* rather than throughput, and the compiler cannot auto-vectorize past that without `-ffast-math`. Measured, this took CPU `sum` at 16 M elements from 7.7 GB/s to **36.9 GB/s**, from 4× behind NumPy to 1.06× ahead of it. It is still single-threaded: PyTorch is 2.8× faster there by spreading the reduction across 20 cores, and a `parallel for` with a `reduction(+:)` clause is the untried next step. `reduce_min_cpu`/`reduce_max_cpu` deliberately do **not** carry an OpenMP pragma of any kind — no `parallel for` (too little work per element for fork/join to pay for itself) and, less obviously, no `#pragma omp simd reduction(min:)/(max:)` either, even though that looks like the natural counterpart to the sum-lane trick above. It was tried and measured: isolated A/B, identical loop body, same `-O3 -march=native -fopenmp` flags, ~1.6× *slower* than the plain scalar loop at 16M elements on GCC 13/aarch64, reproducibly regardless of call order. `-fopt-info-vec-optimized` explains why — GCC already auto-vectorizes the branch-and-select idiom (`if (x < acc) acc = x`) under `-O3` alone, and the explicit `reduction(min:)` clause forces a different, worse lowering on top of an already-vectorized loop rather than improving on it. Left as the plain scalar form as a result. Do not re-add that pragma without re-measuring on the target compiler/architecture first — see [benchmark_report.md §7](benchmark_report.md#7-cpu-elementwise-ops-were-single-threaded--openmp-closes-most-of-it) for the isolated numbers.

## Shape ops, transpose, matmul

These live outside the binary-op macro machinery and each has its own constraints.

**`reshape` / `flatten` / `squeeze` / `unsqueeze`** ([headers/tensor.inl](headers/tensor.inl)) are **deep copies, not views** — `reshape` does `Tensor out(*this)` (the copy ctor allocates and copies) and then rewrites `m_Shape`/`m_Stride` in place. Nothing in the library returns an aliasing view of another tensor's buffer, so unlike NumPy/PyTorch a reshape never lets a write show up through the original. The other three delegate to `reshape`. All are host-side only — no kernel, no stream overload — and `squeeze` of the last remaining axis yields shape `{1}` rather than a scalar.

**`transpose()` is rank-2 only** and throws otherwise; use `permute(axes)` for higher ranks. Both have real CPU and GPU paths ([headers/ops/cpu/transpose_cpu.h](headers/ops/cpu/transpose_cpu.h), [src/ops/kernels/transpose_gpu.cu](src/ops/kernels/transpose_gpu.cu)) and both have `(…, const Stream&)` overloads. `permute` validates axes on the host (length == rank, in range, no duplicates) before dispatching.

`launch_permute` takes the axes as a **host** `const size_t*` and copies them into an `AxesBuf` — a trivially-copyable struct passed **by value** into the kernel parameter block. This is the same rule as `DeviceTensorView`: no device allocation for small per-launch metadata, and no raw pointer members, which would decay to unusable host pointers on the device side.

**`matmul` is 2D-only** in both backends — rank != 2 or mismatched inner dimensions throw from `Tensor::matmul` ([headers/tensor.inl](headers/tensor.inl)) and again inside `matmul_cpu`. It is registered in the legacy dispatch table (`DEFINE_DEVICE_DISPATCH_BINARY_H(matmul, …)`) but `Tensor::matmul` bypasses it and calls `matmul_cpu` / `launch_matmul` directly, like every other stream-aware op. No batching, no broadcasting.

`matmul_cpu` ([headers/ops/cpu/matmul_cpu.h](headers/ops/cpu/matmul_cpu.h)) is `ikj`-ordered with 128-wide L2 tiling and an `omp parallel for` over the independent output rows. The ordering is the load-bearing part and it is not interchangeable: the original `ijk` indexed `rhs(k, j)` down a column, one cache miss per inner iteration, and measured **1.81 GFLOP/s** at 1024³ — 421× off NumPy. Walking `k` in the middle makes both the `rhs` row and the `dst` row sweep contiguously in the innermost `j` loop, so it vectorizes; blocking `i`/`k`/`j` keeps each panel L2-resident. That is worth **68×** (1.81 → 123 GFLOP/s), leaving a 6.3× gap to OpenBLAS at 1024³ and a slight *win* at 128³. It also asserts its operands are contiguous row-major and walks raw pointers rather than paying `compute_flat_index` per access — safe only because nothing in the library hands `matmul` an aliasing or strided view. The remaining gap is register blocking, NEON intrinsics and packed panels; see [benchmark_report.md §6](benchmark_report.md#6-matmul-the-cpu-gap-closed-68-the-gpu-one-remains).

The GPU kernel is 10.3× off cuBLAS and that is a different list: one output element per thread (two shared-memory loads per FMA, so the LDS issue rate is the ceiling), no double buffering, and no tensor cores — `test_benchmarks` prices the last one directly, with fp16 buying only ~1.1× over fp32.

## Python FFI layer

[src/python/openmat_capi.cpp](src/python/openmat_capi.cpp) is the C-ABI boundary, compiled into `OpenMat.so`. Conventions:
- Tensor handles are opaque `void*` to heap `Tensor<T>`; every `om_*_create`/`_copy` must be matched by exactly one `om_*_destroy`.
- Pointer-returning functions return `nullptr` on failure; int-returning ones return non-zero. Both write the exception message into a caller-supplied `char* errbuf, int errbuf_len` (Python passes a 512-byte buffer). No exception crosses the boundary: every entry point is wrapped in `OM_GUARD_PTR` / `OM_GUARD_INT` / `OM_GUARD_VAL`.
- Infallible metadata getters (`rank`, `size`, `shape`, `stride`, `on_cuda`, `device_id`, `dtype`, `itemsize`, `data_ptr`) take no errbuf.

**The per-dtype surface is one body included twice.** [src/python/openmat_capi_impl.inc](src/python/openmat_capi_impl.inc) holds every `om_tensor_<sfx>_*` function; `openmat_capi.cpp` includes it once with `OM_T=float, OM_SFX=float` and once with `OM_T=int, OM_SFX=int`, so `om_tensor_float_add` and `om_tensor_int_add` will not grep as literal definitions (`OM_FN(name)` pastes them). The `.inc` is not in the `file(GLOB src/python/*.cpp)` and is never compiled on its own; it `#error`s if `OM_T`/`OM_SFX` are unset. **Adding a dtype = adding one `#define`/`#include`/`#undef` block** — provided the kernels are instantiated for it (see the `INSTANTIATE_*` macros; `double` has no GPU instantiation, so it cannot be added as-is).

Beyond the tensor families the library exports a dtype-independent runtime API: `om_cuda_device_count`, `om_cuda_is_available`, `om_device_synchronize`, and `om_stream_create/retain/release/destroy/synchronize/handle`.

**Streams are reference-counted at the C boundary** (`StreamBox` in `openmat_capi.cpp`), not in Python. This is deliberate: `cudaMallocAsync` memory must be freed on the stream that produced it, and Python's cyclic collector finalizes a cycle's members — plus everything reachable only from them — in arbitrary order, so a `Stream` object could be torn down before the tensors it still owns memory for. Holding a Python reference is not enough; both attempts at that segfaulted under `gc.collect()`. Each `Tensor` therefore holds an integer handle plus one C-side reference (`om_stream_retain` in `Tensor._wrap`, `om_stream_release` in `__del__`, after the tensor is destroyed). `Stream.close()` is consequently safe while tensors from that stream are alive.

**Float and int32.** `Tensor<double>` and `Tensor<char>` are not exported.

Adding a Python-visible method means three edits: the function in [openmat_capi_impl.inc](src/python/openmat_capi_impl.inc) (once — both dtypes get it), the `ctypes` `restype`/`argtypes` in `_declare_dtype()` in [python/openmat/_clib.py](python/openmat/_clib.py), and the wrapper in [python/openmat/tensor.py](python/openmat/tensor.py). Ops with a `(args, const Stream&)` C++ overload get a `_stream` sibling in the `.inc` and a `stream=None` kwarg in Python.

The Python package is `Tensor` + `Stream` + a `DType` registry ([python/openmat/_dtypes.py](python/openmat/_dtypes.py)); host tensors expose `__array_interface__` (zero-copy `np.asarray`), CUDA tensors `__cuda_array_interface__`. See [python/README.md](python/README.md) for the user-facing surface.

## Benchmarking

Four harnesses, all requiring a **Release** build (`build-release/`) — Debug numbers are meaningless — and a `bench-env` venv holding NumPy and PyTorch, which the library itself does not need:

- [scripts/bench_vs.py](scripts/bench_vs.py) — the main cross-framework table (CPU + CUDA + transfers). `--quick` for a smoke run, `--no-cuda` to skip the device.
- [scripts/bench_rank_sweep.py](scripts/bench_rank_sweep.py) — the same 16 M buffer reshaped to ranks 1–5, which is what verifies the contiguous fast path is actually engaging. `bench_vs.py` only ever uses rank-1 shapes and would not notice a regression here.
- [scripts/bench_omp.py](scripts/bench_omp.py) — run once per `OMP_NUM_THREADS` value; the 1-vs-20 delta is the OpenMP A/B, and a `1.00×` row is how you confirm an op is *not* parallelized (`min`/`max`, `sum`, `apply`/`apply_binary`).
- `OPENMAT_HOST_CACHE_BYTES=0` — restores the pre-`HostPool` allocator, the A/B behind §3.

Two traps that have already produced wrong conclusions in the reports:

- **`bench_vs.py`'s `transfer/*` rows at 16 M are not transfer measurements.** They run last, after the process has accumulated `HostPool`, `PinnedHostPool`, PyTorch's CUDA caching allocator and every live operand from the CUDA sweep; on a unified-memory part that pressure lands on the copy. The row reported OpenMat's 64 MB D2H at 39.8 ms while a fresh process measures **1.136 ms / 59.1 GB/s, identical to PyTorch**. The tell is that PyTorch's own D2H degrades alongside it in the same run. Measure transfers in a dedicated process.
- **Re-measure before diagnosing.** A block-size fix once moved GPU `add` from 159 to 220 GB/s without the report being re-run, and the next round of work was aimed at a bottleneck that no longer existed. Every number in [benchmark_report.md](benchmark_report.md) carries the date of the run that produced it for this reason.

## Reference docs

- [README.md](README.md) — measured stream benchmarks (RTX 4060 + GB10) and the rationale behind each design decision.
- [benchmark_report.md](benchmark_report.md) — OpenMat vs NumPy vs PyTorch, with the root-cause analysis behind each gap.
- [stream_perf_report.md](stream_perf_report.md) — raw `test_stream_perf` output, plus `test_benchmarks` and `test_stress`. The **GB10 appendix is the current source** (re-measured 2026-09-04, medians of 9 runs); the RTX 4060 sections above it are a 2024 snapshot of a different code state and should not be quoted as current. Nine runs rather than four because the suite times a single un-warmed run per variant and the spread is wide — the sequential chain alone produced single runs from 0.98× to 1.16×.
- [docs/fused_operations.md](docs/fused_operations.md) — fusion design walkthrough.
- [docs/roadmap.md](docs/roadmap.md) — planned work with a done/not-done priority table at the end (written in Italian).
