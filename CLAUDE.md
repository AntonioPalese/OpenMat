# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Requirements: NVIDIA GPU, CUDA Toolkit ≥ 11.2 (for `cudaMallocAsync`), CMake ≥ 3.24 (for `CMAKE_CUDA_ARCHITECTURES=native`), C++17/CUDA 17 compiler, OpenMP (`find_package(OpenMP REQUIRED)` in `CMakeLists.txt`; bundled as `libgomp` with a stock GCC install, nothing extra to install on most systems). Verified building and passing all 18 suites on CUDA 13.0 / GCC 13.3 / CMake 3.28 / GB10 (sm_121). 15 of them are correctness suites; the other three are timing/soak suites (see Tests).

```bash
# Full clean rebuild: Debug, serial make (also refreshes compile_commands.json in the repo root)
./compile.sh

# Incremental Release build for benchmarking (build-release/ is the conventional dir)
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release && cmake --build build-release -j
```

`compile.sh` does `rm -rf build` every time and runs a plain `make` with no `-j`. For an incremental rebuild, run `cmake --build build -j`.

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
cd build && ctest -E "test_benchmarks|test_stream_perf|test_stress"   # correctness suites only
./build/tests/test_arithmetic           # one suite, per-test output
./build/tests/test_arithmetic --gtest_filter="TensorArithmetic.CPUOperations"
```

`test_benchmarks`, `test_stress`, and `test_stream_perf` are timing/soak suites, not correctness suites — they are slow and their numbers are meaningless in a Debug build. `StreamPerf.ParallelFanOut` in particular asserts wall-clock against wall-clock and goes red under load. The GPU CI job excludes `test_benchmarks` and `test_stream_perf` (`ctest -E`), but `test_stress` **does** run there and can fail CI.

**Every test that touches the device starts with `OM_REQUIRE_CUDA();`** ([tests/test_helpers.h](tests/test_helpers.h)) — a `cudaGetDeviceCount` check that `GTEST_SKIP`s instead of failing. That is what makes `ctest` green on a machine with no GPU (157 skipped, 110 host-side tests run, as of 2026-09-29) and is what the CPU-only CI job relies on; a new GPU test without it turns that job red. The Python equivalents are the `requires_cuda` marker and the `device` fixture in [python/tests/conftest.py](python/tests/conftest.py).

## Debugging asynchronous kernel errors

`OPENMAT_DEBUG_SYNC=1` localizes an async kernel error to its launching call site; see the `debug-cuda-async` skill for the walkthrough and the `compute-sanitizer` follow-up.

**Every kernel launch must be followed by `CUDA_CHECK_LAUNCH(kernel_name, stream)`** ([headers/cuda_defines.cuh](headers/cuda_defines.cuh)), not the older bare `CUDA_CHECK` — that one has no way to know the stream or the kernel, so it can only do the synchronous check. In the rank-switching launcher macros the name is carried in a local `const char* om_kernel` assigned in each branch, with one check after the `switch`. The implementation is deliberately non-`inline`, in [src/cuda_debug.cpp](src/cuda_debug.cpp): both switches are preprocessor-conditional, and an inline definition would let a consumer built with different settings supply a conflicting second definition (ODR violation, linker silently picks one).

## CI

[.github/workflows/ci.yml](.github/workflows/ci.yml), on every push and PR. Two jobs:

- **cpu** — toolkit, no driver, no device. Its job is the full compile+link — the macro machinery and the mandatory explicit instantiations only fail there — plus the host-side tests.
- **gpu** — a self-hosted runner. Release build, the correctness suites, the Python suite, then `compute-sanitizer --tool memcheck --leak-check full`. memcheck is the only thing that reports a stream-ownership violation near the call site instead of as an illegal access in an unrelated kernel later on.

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

Python suites: `test_tensor.py` (the original surface), `test_tensor_api.py` (metadata, indexing, shape/fused ops, buffer protocols), `test_dtypes.py`, `test_streams.py`, `test_inplace.py`, `test_broadcast.py`, `test_views.py`, `test_dlpack.py` (NumPy half always runs; the PyTorch half skips when `torch` is not importable, which is the case in the CPU CI job). [python/tests/conftest.py](python/tests/conftest.py) provides a `device` fixture that runs a test on both backends and a `requires_cuda` marker.

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

**When adding an op, implement the `(args, out, const Stream&)` overload and make everything else a one-line delegate.** Doing it the other way round breaks the single-source-of-truth invariant the whole `tensor.inl` is built on. See "In-place ops and caller-provided destinations" below for the three forms and which one is the real implementation.

### Dispatch: two paths, one of them mostly dead

[headers/kernel_launcher.h](headers/kernel_launcher.h)/[.inl](headers/kernel_launcher.inl) still define the macro-generated dispatch machinery:
- `DEFINE_DEVICE_DISPATCH_BINARY_H(OP, CPU_FUNC, CUDA_FUNC)` — declares `OP_dispatch<DEVICE_TYPE, T>` structs.
- `DEFINE_DEVICE_DISPATCH_BINARY_INL(OP)` — defines the free function `_OP(lhs, rhs, dst, DEVICE_TYPE)` that switches at runtime.
- `DEFINE_DEVICE_DISPATCH_UNARY_H/INL` — same for tensor⊕scalar ops.

**But `Tensor<T>` no longer goes through them at all.** Since the stream refactor, [headers/tensor.inl](headers/tensor.inl) branches on `device_type()` itself and calls `add_cpu(...)` / `launch_add(..., s.get())` directly, because the `_dispatch` structs have no `cudaStream_t` parameter. `fill` was the last holdout and is now `fill_(value, s)` for the same reason, so the `_dispatch` machinery has **no callers left**. Treat it as legacy: adding a new op means wiring `tensor.inl` directly, and adding a dispatch registration only if you actually want the stream-less free function.

Effective data flow for `a + b`:

```
Tensor<T>::operator+()
  → Tensor<T>::add(rhs, Stream::default_stream())          [tensor.inl]
    → allocates the result, then
      Tensor<T>::add_out(rhs, out, stream)                 [tensor.inl — the real body]
        → add_cpu(...)  or  launch_add(..., stream)        [ops/cpu/ or ops/kernels/]
          → flat CPU loop  or  contiguous fast path / rank-specialized CUDA kernel
```

`a.add_(b)` enters the same chain one level down, at `add_out(b, *this, stream)`.

### Broadcasting

Tensor-tensor elementwise ops (`add`/`sub`/`mul`/`div`, `apply_binary`, the `fused_*` helpers built on it, and every `_out`/`_` form) follow NumPy broadcasting. It is done entirely with **zero strides**, in [headers/broadcast.h](headers/broadcast.h): `_check_operand` returns `detail::broadcast_shapes(lhs, rhs)` (which throws `not broadcastable`, and throws for a result rank above `MAX_RANK`), and the `_out` body turns each operand into `detail::expand_to(shape, stride, out_shape)`. That is a view with the output's shape and stride 0 on every axis the operand is really missing or has as extent 1. Every kernel indexes through strides, so no data is copied and no kernel knows broadcasting exists. The `ExpandedLayout` arrays are inline and local to the `_out` call, which outlives the launch; the launch copies them into `DeviceTensorView` by value.

Consequences:
- When the shapes are equal, `expand_to` returns the operand's own strides, so a same-shape call is bit-for-bit the pre-broadcast call and keeps the contiguous fast path. Verified in a Release A/B: the rank sweep and CPU `add` at 16 M are unchanged.
- A broadcast operand's view is not `is_contiguous()`, so the fast path declines it and it runs on the rank-specialized kernels (GPU) or the row-wise strided loop in `detail::binary_elementwise_cpu` ([headers/ops/cpu/broadcast_cpu.h](headers/ops/cpu/broadcast_cpu.h)), which is also `apply_binary`'s CPU path now.
- The binary launchers check `same_shape`, not `match`: `match` also compares strides, which an expanded view never shares with `dst`.
- In place follows PyTorch: `x.add_(bias)` works, `bias.add_(x)` throws, because the result shape must equal the destination's. A broadcast operand that shares storage with `out` (`m.add_(m[0])`) is refused by the aliasing check (see "Views" below): its 0-stride reads would see elements already written.
- GPU broadcast is at PyTorch parity for the common cases: `(4096,4096)+(4096,)` runs in 541 µs against PyTorch's 544, and `+(4096,1)` in 546 against 542 (GB10, Release, one framework per process). A rank-5 broadcast still goes through the 64-bit `_nd` kernel, at 614 against 546. CPU broadcast is 4.2× ahead of NumPy. Two fixes got it there, and neither was specific to broadcasting (see `DeviceTensorView` and "The device pool" below). A kernel redesign turned out not to be needed: a `(256,1)` block, 4 elements per thread and a flat 1-D layout were all measured, and none of them moved the number.

### In-place ops and caller-provided destinations

Every op exists in three forms, and only one of them is a real implementation:

```cpp
auto c = a.add(b, s);        // allocates a result, then calls add_out
a.add_out(b, out, s);        // ← the body: writes into a destination that exists
a.add_(b, s);                // == a.add_out(b, a, s)
```

`add_out` returns `Tensor&` (the destination), so calls chain; `add_` returns `*this`. The no-stream spelling of each is a one-line delegate with `Stream::default_stream()`. `operator+=`/`-=`/`*=`/`/=` are delegates to the `_` family.

The families: `add`/`sub`/`mul`/`div` (tensor and scalar), `apply`, `apply_binary`, `relu`, `sigmoid` and `fill_` have all three forms. `matmul`, `transpose` and `permute` have `_out` but **no** in-place form — their kernels read elements they do not write, so a destination sharing a buffer with an operand would read values already overwritten. `_check_alias_none` rejects that at the call site instead of returning a plausible wrong answer.

Why the elementwise family *can* alias: the CPU loop, the contiguous GPU fast path and the rank-specialized kernels all read index i and write index i through each operand's own strides, so an operand that is *the same view* as `dst` (same start, shape and strides) is exactly as correct as a separate buffer. Any other overlap is refused — see "Views" below.

Two consequences worth knowing:

- **The contiguous kernels dropped `__restrict__`** ([headers/ops/kernels/contiguous.cuh](headers/ops/kernels/contiguous.cuh)) — in-place calls them with `dst` equal to a source, which is precisely the aliasing the qualifier promises does not happen. Measured, it cost nothing: `add` at 16 M went 230.2-233.4 → 228.6-230.7 GB/s, `relu` 232.9-234.6 → 233.4-235.5. The read-only path comes from the explicit `__ldg` in `device_load`, not from restrict, and each pointer is touched once per thread. Re-measure before putting it back.
- **A destination that is not freshly allocated carries a stream caveat the allocating path does not.** `cudaMallocAsync` memory is stream-ordered, so enqueueing into `out` on a stream other than the one `out` was allocated on is only correct once the caller has ordered the two (an event, or a synchronize). `Tensor` cannot check that, so it is documented, not enforced — unlike the ownership invariant below, which it does enforce by keeping the allocation stream in the `Storage`.

The `_out` core is also where operand validation now lives: device and broadcast compatibility are checked before dispatch, so a GPU `add` between incompatible shapes throws instead of running off the end of a buffer the way it used to.

`scripts/bench_inplace.py` is the harness; [benchmark_report.md §9](benchmark_report.md#9-every-op-allocated-its-own-result--in-place-and-out-forms-added) has the numbers. The short version: 1.2-1.6× below ~64 K on both backends where allocation is a large fraction of the op, shrinking to 1.04× at 16 M where the kernel dominates. The memory argument does not shrink — `tests/test_inplace.cpp` and `python/tests/test_inplace.py` assert `data_ptr` is unchanged across a 100-step loop, which is the property a correct-but-reallocating implementation would fail while passing every value check.

### Memory and views

**`Tensor<T>`** ([headers/tensor.cuh](headers/tensor.cuh), [headers/tensor.inl](headers/tensor.inl)) — a (storage, offset, shape, stride) tuple: `std::shared_ptr<Storage<T>> m_Storage`, `m_Offset`, and `m_Data` cached as `storage->data() + offset`. Copy is a **deep copy into a fresh contiguous buffer**, whatever the source's strides (that is `clone()`); views come only from the methods that say so (see "Views" below). Move transfers the storage reference and nulls the source. There is a private `Tensor(shape, device, Stream)` constructor used by the stream overloads so a result tensor is allocated and freed on the stream that produced it.

**`Storage<T>`** ([headers/storage.h](headers/storage.h)) — the buffer: `T*`, element count, `Device`, the `unique_ptr<Allocator<T>>` and the `Stream` it was allocated on. Non-copyable, owned through `shared_ptr` (built with `make_shared`, so one extra heap block per tensor; measured, the small-size CPU ops did not get slower once the OpenMP branch below was fixed).

**`Allocator<T>` / `AllocatorFactory<T>`** ([headers/allocator.h](headers/allocator.h), [.inl](headers/allocator.inl)) — `CpuAllocator` (host block cache/memcpy) and `GpuAllocator` (cudaMalloc/cudaFree/cudaMemcpy). The base class declares `allocate_async`, `deallocate_async`, `copy_async`, `copy_host_to_device_async`, `copy_device_to_host_async` with **synchronous default implementations**, so a subclass overrides only what it can actually do async. `GpuAllocator` uses `cudaMallocAsync`/`cudaFreeAsync` under `#if CUDART_VERSION >= 11020`.

**Host memory is recycled, not returned to the OS.** `CpuAllocator::allocate/deallocate` go through `om::detail::HostPool` ([headers/host_pool.h](headers/host_pool.h)), a process-wide free list keyed by size class. Plain `malloc` was measurably the dominant cost of every host-side op: above glibc's 128 KB `MMAP_THRESHOLD` each allocation is a fresh `mmap`, so an out-of-place op page-faults its whole output buffer before computing anything — 16384 faults per 64 MB result. Recycling a block keeps its pages mapped and took `add` at 16 M elements from 19.4 ms to 4.5 ms and a 64 MB `Tensor::cpu()` from 18.2 ms to 1.14 ms (the D2H case pays the faults *inside* `cudaMemcpy`, which is why the allocator showed up as a transfer problem).

Details that matter if you touch it: requests round up to a size class (8 per octave, ≤ 12.5 % overshoot) so the number of free lists stays bounded; each block carries a 64-byte header holding its class, so `deallocate` needs no side table and the pointer handed out is 64-byte aligned rather than malloc's 16; the cache is capped (256 MB by default, `OPENMAT_HOST_CACHE_BYTES` overrides, `0` disables recycling and restores the old behaviour — useful for A/B measurement); and the singleton is deliberately never destroyed, because a `Tensor` with static storage duration would otherwise free into a destroyed pool. Host pointers must therefore never be freed with bare `std::free`, and host memory must never be allocated with bare `malloc` and handed to a `Tensor`.

**Pinned (page-locked) host memory is opt-in, not automatic.** `PinnedCpuAllocator<T>` (subclasses `CpuAllocator<T>`, overrides only `allocate`/`deallocate`) allocates through `om::detail::PinnedHostPool` — the same size-class recycling as `HostPool`, but `cudaHostAlloc`/`cudaFreeHost` instead of `malloc`/`free`, capped separately (`OPENMAT_PINNED_CACHE_BYTES`, default 64 MB) because page-locking is one to two orders of magnitude slower than a pageable allocation, so recycling matters even more here. Nothing decides on its own which host tensors will cross the bus — a `Tensor` gets `PinnedCpuAllocator` only via `Tensor::pinned(shape)` (an explicit request, for a buffer known to be a repeated H2D *source*) or as the destination `Tensor::to()` allocates for a device-to-host copy, where it isn't a guess: that exact buffer's only purpose is to receive that exact copy. `device_type()` still reports `CPU` either way — there is no third `DEVICE_TYPE`; pinned-ness is purely which allocator subclass `m_Allocator` holds, queryable via `Tensor::is_pinned()` (a `dynamic_cast`). Both call sites already require a working CUDA driver, so unlike `HostPool` this pool is never touched by the CPU-only CI job — it still has to compile there (against the stub `libcuda.so`), but nothing exercises it. One consequence of using a CUDA-tracked allocation API for a "never destroyed" singleton: `compute-sanitizer --leak-check full` (which the gpu CI job runs with `--error-exitcode 1`) would flag every still-cached block as leaked at exit, unlike `HostPool`'s plain-`malloc` cache, which the sanitizer can't see. `PinnedHostPool`'s constructor registers an `atexit` hook that empties the free list on the way out; it only calls `release_all()`, never destroys the pool object, so it doesn't reopen the destruction-order hazard the singleton pattern exists to avoid. See [benchmark_report.md §3](benchmark_report.md#3-the-cpu-gap-above-128-kb-was-the-allocator-not-the-loop--fixed) for what this does and does not move on the reference hardware.

**The device pool keeps its memory.** On the first stream-ordered allocation on each device, `detail::ensure_device_pool_configured()` ([src/device_pool.cpp](src/device_pool.cpp), out of line for the same ODR reason as `cuda_debug.cpp`) raises that device's default `cudaMallocAsync` pool release threshold to unlimited. The CUDA default is 0, meaning the pool trims every unused byte at every synchronize, and the synchronous API synchronizes after every op. So a result that was dropped and then re-allocated at the same size had to map fresh physical pages each time. That cost ~150 µs on a 64 MB op in a plain loop, and 3.2 ms once the caller also synchronized, which is more than the kernel. PyTorch's caching allocator never returns memory, and this gives the pool the same policy. `OPENMAT_CUDA_POOL_RELEASE_BYTES` overrides the threshold; `0` restores CUDA's default and is the A/B. `compute-sanitizer --leak-check full` stays clean because pool reservations are not allocations. The setting applies to the device's default pool, so it also affects any other `cudaMallocAsync` user in the process.

**Stream-ownership invariant:** `cudaMallocAsync` memory belongs to a stream-ordered pool and must be freed on the stream it was allocated on. That is why the `Storage` keeps the allocation stream and its destructor calls `deallocate_async(data, stream)`: whichever view of a tensor dies last, the free lands on the original stream, never on the view's. `Tensor::stream()` reports the storage's stream. The caller still has to keep that stream alive as long as any view of the storage — which is why the Python layer takes one C-side stream reference per view (`Tensor._wrap_view`), not just per allocation. Breaking this shows up as an illegal memory access far from the real call site.

**`TensorView<T>`** ([headers/tensor_view.cuh](headers/tensor_view.cuh)) — non-owning host-side view (pointer + shape/stride pointers + rank), `__host__`-only. Converted with `.as_device_tw()` at launch.

**`DeviceTensorView<T>`** ([headers/device_tensor_view.cuh](headers/device_tensor_view.cuh)) — non-owning device-side view. Shape and stride are **fixed inline arrays** (`size_t shape[MAX_RANK]`, `MAX_RANK = 8`) filled from host at construction, so the struct is trivially copyable and passed **by value** into the kernel parameter block. There is deliberately no device allocation here — an earlier design cudaMalloc'd shape/stride per view (6 allocations per binary op). Do not reintroduce pointer members: raw arrays passed as kernel arguments decay to host pointers on the device side. Rank > 8 trips the constructor `assert`. `operator()` computes the offset as a fold over the index pack (`offset_of`). **Do not go back to copying the indices into a local array and looping over `rank`.** `rank` is a runtime field, so the compiler places that array in local memory: every rank-specialized kernel had a 16-byte stack frame and paid a local store and reload per element. That alone held broadcast `add` at 153 GB/s, and the fold version reaches 250 GB/s, which is the copy ceiling. `cuobjdump --dump-resource-usage build/OpenMat.so | grep STACK` is the check; only `permute_kernel` still has a frame, from its own index arrays.

**`Device`** ([headers/mat_utils.h](headers/mat_utils.h)) — `m_Id`, `m_Str`, `m_Dt`. Constructible from `(id, DEVICE_TYPE)` or a string like `"cuda:0"`.

### Rank-specialized CUDA kernels

[headers/ops/kernels/binary_op_macros.cuh](headers/ops/kernels/binary_op_macros.cuh): `DEFINE_BINARY_OP_LAUNCH(OP)` generates `launch_OP(lhs, rhs, dst, cudaStream_t)` which switches on `lhs.rank` and picks a kernel with a rank-tuned grid/block layout (1D `dim3(16)`, 2D `dim3(16,16)`, 3D/4D `dim3(8,8,8)`). Rank ≥ 5 falls back to `OP_kernel_nd`, a flat 1D kernel reconstructing multi-indices from a linear index. `DEFINE_BINARY_OP_LAUNCH_FRW_DEC(OP)` emits explicit instantiations for `float`, `int`, `char`, `float16_t`.

**The `_nd` kernel is also the overflow path, not just the rank ≥ 5 path.** `gridDim.y` and `gridDim.z` are capped at 65535 (only `gridDim.x` reaches 2^31-1), so a leading axis large enough overflows the rank-specialized layout — the rank-4 launcher sets `blocks.z = shape[0]`, the rank-3 one `(shape[0] + 7) / 8`. Past that the launch fails synchronously with `invalid configuration argument`, which surfaces to the caller as a generic CUDA error naming nothing useful. Every rank-specialized launcher therefore computes its grid extents as `size_t` and gates the launch on `om::detail::grid_fits(gx, gy, gz)` ([headers/cuda_defines.cuh](headers/cuda_defines.cuh)), falling through to the flat `_nd` kernel when they do not fit. The extents are checked *before* being narrowed into a `dim3` — its members are `unsigned int`, so building one first could truncate an oversized extent into a plausible small one. The same guard is in the unary launcher, `launch_fill`, `launch_apply_op` and `launch_apply_binary_op`; a new rank-switching launcher needs it too. The binary `_nd` kernels (`DEFINE_BINARY_OP_KERNEL_ND`, `apply_binary_op_nd`) compute **one offset per operand**, last axis fastest. They used to reuse one operand's offset for all three buffers, which was only correct while every stride matched, and broadcasting breaks that.

**Every elementwise launcher tries a contiguous fast path first.** [headers/ops/kernels/contiguous.cuh](headers/ops/kernels/contiguous.cuh) — the rank-specialized layouts only keep a warp's 32 lanes contiguous in memory at rank 1. A rank-2 `dim3(16,16)` block gives each warp two disjoint runs of 16 elements, a rank-3 `dim3(8,8,8)` block four runs of 8: 64- and 32-byte requests against a 128-byte line. Measured on the reference GB10, `add` over 16 M floats ran at 230 GB/s at rank 1, 206 at rank 2, 141 at rank 3, 104 at rank 4 and 26 at rank 5 (`_nd`, which also recomputes a multi-index per element) — same traffic, same op. Since almost every tensor is contiguous row-major (every fresh allocation, and every reshape or leading-axis slice of one), the shape carries nothing the kernel needs, so the fast path indexes the buffer linearly and gives every rank the rank-1 layout: all of them now measure 228-233 GB/s. `launch_add`/`launch_sub`/`launch_mul`/`launch_div`, the `_k` scalar family, `launch_apply_op`, `launch_apply_binary_op` and `launch_fill` all take it.

`TensorView::is_contiguous()` is what gates it — the launchers ask rather than assume, so a strided view falls back to the existing rank-specialized kernels instead of silently reading the wrong elements. That is the whole reason the guard is a runtime check and not a comment, and broadcasting already relies on it: an expanded operand carries 0 strides and is declined here (see "Broadcasting" above).

Two things about it are counter-intuitive and were measured, not assumed:

- **The pack width is 4 bytes per thread, not 16.** `float4` vector loads are the standard advice and they are *slower* here — one 16-byte `float4` per thread drops `add` to 216 GB/s against 235 for one scalar `float`, because the launch loses the thread-level parallelism that keeps the memory pipeline full. A grid-stride loop capped at a few waves per SM costs a further 8-12 %; at the exact block count its loop body runs once and buys nothing over a bounds check. Neither is used. What does matter is that no thread moves *less* than 4 bytes: a warp of `char` lanes requests 32 bytes and reaches only 193 GB/s even at rank 1. Hence `pack_width<T> = 4 / sizeof(T)` — 1 for `float`/`int`, 2 for `float16_t`, 4 for `char`, which takes `char` to 236 GB/s. The pack is punned through a `unsigned int` with `memcpy`, not a union or a member-wise struct copy: `float16_t` has a user-provided constructor, so a union of it is ill-formed and a member-wise copy is free to lower to two 2-byte accesses, which is exactly what the pack exists to avoid.
- **A size that is not a multiple of the pack width leaves a tail**, picked up by block 0 after the packed loop. `test_contiguous` runs every dtype at sizes covering all four residues mod 4 precisely because nothing else in the suite would notice the tail being dropped.

Block size is 256, as elsewhere in the library. 512 and 1024 buy 2-3 % once the working set passes L2 and lose up to 35 % below it, where occupancy rather than bandwidth is the limit. Re-measure before changing it.

The kernel definitions and the launch statements sit behind `#if defined(__CUDACC__)`: `tensor.cuh` pulls this header into plain `.cpp` translation units (the Python C-ABI layer among them), where `__global__` expands to nothing and `blockIdx` does not exist. A new elementwise launcher that wants the fast path calls `om::detail::launch_contiguous_binary/unary/fill`, which return the launched kernel's name for `CUDA_CHECK_LAUNCH` or `nullptr` when they decline — declining is not an error, it is how an empty tensor, an unaligned buffer or an unrepresentable grid keeps the old, always-correct path. The four generated op families need one more piece: `DEFINE_BINARY_OP_FUNCTOR_H` / `DEFINE_UNARY_OP_FUNCTOR_H` turn the op's expression into a functor type, because the rank-specialized kernels take it textually but the fast path is generic over the operation. See [benchmark_report.md §8](benchmark_report.md#8-elementwise-kernels-ignored-contiguity--every-rank-now-runs-at-rank-1-speed).

**Adding a new binary op:** see the `add-binary-op` skill for the file-by-file checklist.

**CPU binary/unary elementwise ops parallelize above a size threshold.** `DEFINE_BINARY_OPS_CPU` ([headers/ops/cpu/binary_op_macros.h](headers/ops/cpu/binary_op_macros.h)) and `DEFINE_UNARY_OPS_CPU` ([headers/ops/cpu/unary_op_macros.h](headers/ops/cpu/unary_op_macros.h)) — the CPU side of `add`/`sub`/`mul`/`div`, both tensor⊕tensor and tensor⊕scalar — delegate to `detail::binary_elementwise_cpu` / `detail::unary_elementwise_cpu` ([headers/ops/cpu/broadcast_cpu.h](headers/ops/cpu/broadcast_cpu.h)), whose contiguous loop is `#pragma omp parallel for schedule(static)` above 65536 elements. The unary helper also carries `apply`/`relu`/`sigmoid` on the CPU, which were a single-threaded loop until the views work routed them through it (`relu` at 1 M: 69 → 12 µs). The threshold is an **explicit `if (total > 65536)` branch around two loops, not an `if()` clause on the pragma**: GCC still calls into the OpenMP runtime for a region whose `if()` is false, and that call alone was worth 0.28 µs at 1 K elements — replacing the clause took `add`/`mul` at 1 K from 1.92 to 1.65 µs (A/B, 2026-09-29). The strided (row-walk) paths keep the `if()` clause; they are not the small-size hot path. `_Pragma` rather than `#pragma` keeps the spelling usable inside a macro. Below the threshold the loop is measurably untouched (within ~2% of the single-thread time — no fork/join tax paid on the hot path for small tensors); above it, `add`/`sub`/`mul`/`div` measured 1.7–11.6× faster on a 20-thread reference machine (largest at 1 M, where the working set is L2-resident and the scalar loop rather than memory was the ceiling; ~2× at 16 M, which is the memory system talking) and beat NumPy outright by 2.9–3.1× at 16 M. See [benchmark_report.md §7](benchmark_report.md#7-cpu-elementwise-ops-were-single-threaded--openmp-closes-most-of-it) for the numbers. `matmul_cpu` ([headers/ops/cpu/matmul_cpu.h](headers/ops/cpu/matmul_cpu.h)) is parallelized the same way (`#pragma omp parallel for` over the outer row loop, on top of `ikj`-order/L2-tiled inner loops) and predates this.

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

`Tensor<T>` surface: `apply(op[, stream])`, `apply_binary(rhs, op[, stream])`, `scale_shift`, `shift_scale`, `relu`, `sigmoid`, `fused_add_mul`, `fused_sub_mul`, `fused_mul_add`, `fused_div_add`, plus the `_out` and `_` forms of `apply`, `apply_binary`, `relu` and `sigmoid`. `apply` and `apply_binary` both have a real CPU loop branch — the CUDA-only limitation noted in [docs/roadmap.md](docs/roadmap.md) §4.2 has been fixed, and `apply_binary` gained the stream overload §4.4 was asking for when it was rebuilt on `apply_binary_out`. The fused `fused_*` helpers have no dedicated in-place spelling: `apply_binary_(rhs, BinaryCompose<…>{…})` with the same functor is the way to get one.

## Reductions

GPU: two-phase shared-memory tree reduction + warp shuffle (`__shfl_down_sync`) — `launch_reduce_sum/min/max` in [headers/ops/kernels/reduce_gpu.cuh](headers/ops/kernels/reduce_gpu.cuh). CPU: [headers/ops/cpu/reduce_cpu.h](headers/ops/cpu/reduce_cpu.h). Exposed as `.sum()`, `.mean()`, `.min()`, `.max()`; these are synchronous and return a host scalar (no stream overloads).

`reduce_sum_cpu` splits the accumulation across 8 independent lanes (a source-level restructuring, commented inline) to break the loop-carried FP dependency chain — a single accumulator runs at one add per FP *latency* rather than throughput, and the compiler cannot auto-vectorize past that without `-ffast-math`. Measured, this took CPU `sum` at 16 M elements from 7.7 GB/s to **36.9 GB/s**, from 4× behind NumPy to 1.06× ahead of it. It is still single-threaded: PyTorch is 2.8× faster there by spreading the reduction across 20 cores, and a `parallel for` with a `reduction(+:)` clause is the untried next step. `reduce_min_cpu`/`reduce_max_cpu` deliberately do **not** carry an OpenMP pragma of any kind — no `parallel for` (too little work per element for fork/join to pay for itself) and, less obviously, no `#pragma omp simd reduction(min:)/(max:)` either, even though that looks like the natural counterpart to the sum-lane trick above. It was tried and measured: isolated A/B, identical loop body, same `-O3 -march=native -fopenmp` flags, ~1.6× *slower* than the plain scalar loop at 16M elements on GCC 13/aarch64, reproducibly regardless of call order. `-fopt-info-vec-optimized` explains why — GCC already auto-vectorizes the branch-and-select idiom (`if (x < acc) acc = x`) under `-O3` alone, and the explicit `reduction(min:)` clause forces a different, worse lowering on top of an already-vectorized loop rather than improving on it. Left as the plain scalar form as a result. Do not re-add that pragma without re-measuring on the target compiler/architecture first — see [benchmark_report.md §7](benchmark_report.md#7-cpu-elementwise-ops-were-single-threaded--openmp-closes-most-of-it) for the isolated numbers.

## Shape ops, transpose, matmul

These live outside the binary-op macro machinery and each has its own constraints.

### Views

A view shares its base's `Storage`: a write through either one shows up in the other, and the buffer lives until the last of them is destroyed. The rules follow PyTorch:

- **`reshape` / `flatten`** return a view when the tensor is contiguous and a reshaped **copy** otherwise (only a contiguous tensor can be re-read with row-major strides for an arbitrary new shape).
- **`squeeze` / `unsqueeze` / `slice(axis, start, stop, step)` / `select(axis, index)`** are always views. `slice` clamps like a Python slice and needs `step > 0` — strides are `size_t`, so there are no negative strides and a view's first element is always its lowest address. `select` and `squeeze` of the last remaining axis yield shape `{1}`, not a scalar.
- **`contiguous()`** is a view when already contiguous, a copy otherwise; **`clone()`** / the copy constructor always copy. **`copy_(src)`** writes `src` into a view, broadcasting it.
- **`transpose` / `permute` are still copies** (their kernels and `_out` forms are unchanged): there is no stride-permuting view.

What each op does with a strided (non-contiguous) view:
- **Elementwise** (arithmetic, scalar, `apply`, `apply_binary`, `fill_`, `copy_`) — reads and writes through strides on both backends; the contiguous fast paths decline and the stride-aware paths take over. Both GPU `_nd` kernels (unary and `apply`) used to reuse one offset for source and destination and their launchers compared strides (`match()`); both now compute one offset per operand and compare shapes only.
- **Reductions** and **CPU `matmul`** walk one flat run, so a strided input goes through `contiguous()` first. **Transfers** (`to`/`cpu`/`cuda`, `copyToHost`) likewise.
- **`matmul_out` / `transpose_out` / `permute_out`** refuse a non-contiguous destination (`permute_kernel` writes `dst[idx]` flat).

**Aliasing is checked on memory ranges, not pointers.** `detail::views_overlap` ([headers/broadcast.h](headers/broadcast.h)) compares each view's `[first, last]` element span. `_check_alias_elementwise` allows an operand that overlaps `out` only when it is the very same view (`detail::same_layout`: start, shape, strides — the in-place case); a shifted slice of the same storage, or a broadcast operand that is also written, throws. Disjoint slices of one storage are fine. The check is conservative: two interleaved views (even and odd columns) are refused although their elements are disjoint. `_check_alias_none` (matmul/transpose/permute) refuses any overlap.

An empty view (a slice with `start == stop`) is legal and has a null data pointer; `_check_out` therefore tests for a moved-from tensor by its storage, not by its pointer, and every `_out` body returns early on an empty destination.

**`transpose()` is rank-2 only** and throws otherwise; use `permute(axes)` for higher ranks. Both have real CPU and GPU paths ([headers/ops/cpu/transpose_cpu.h](headers/ops/cpu/transpose_cpu.h), [src/ops/kernels/transpose_gpu.cu](src/ops/kernels/transpose_gpu.cu)) and both have `(…, const Stream&)` overloads. `permute` validates axes on the host (length == rank, in range, no duplicates) before dispatching.

`launch_permute` takes the axes as a **host** `const size_t*` and copies them into an `AxesBuf` — a trivially-copyable struct passed **by value** into the kernel parameter block. This is the same rule as `DeviceTensorView`: no device allocation for small per-launch metadata, and no raw pointer members, which would decay to unusable host pointers on the device side.

**`matmul` is 2D-only** in both backends — rank != 2 or mismatched inner dimensions throw from `Tensor::matmul` ([headers/tensor.inl](headers/tensor.inl)) and again inside `matmul_cpu`. It is registered in the legacy dispatch table (`DEFINE_DEVICE_DISPATCH_BINARY_H(matmul, …)`) but `Tensor::matmul` bypasses it and calls `matmul_cpu` / `launch_matmul` directly, like every other stream-aware op. No batching, no broadcasting.

`matmul_cpu` ([headers/ops/cpu/matmul_cpu.h](headers/ops/cpu/matmul_cpu.h)) is `ikj`-ordered with 128-wide L2 tiling and an `omp parallel for` over the independent output rows. The ordering is the load-bearing part and it is not interchangeable: the original `ijk` indexed `rhs(k, j)` down a column, one cache miss per inner iteration, and measured **1.81 GFLOP/s** at 1024³ — 421× off NumPy. Walking `k` in the middle makes both the `rhs` row and the `dst` row sweep contiguously in the innermost `j` loop, so it vectorizes; blocking `i`/`k`/`j` keeps each panel L2-resident. That is worth **68×** (1.81 → 123 GFLOP/s), leaving a 6.3× gap to OpenBLAS at 1024³ and a slight *win* at 128³. It also asserts its operands are contiguous row-major and walks raw pointers rather than paying `compute_flat_index` per access — safe because `matmul_out` passes a strided operand through `contiguous()` first and refuses a strided destination. The remaining gap is register blocking, NEON intrinsics and packed panels; see [benchmark_report.md §6](benchmark_report.md#6-matmul-the-cpu-gap-closed-68-the-gpu-one-remains).

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

The in-place and destination-provided families cross the boundary as **int-returning** functions (`om_tensor_float_add_inplace`, `..._add_out`, `..._add_scalar_inplace`, …), not handle-returning ones: nothing is allocated and no ownership changes hands, so there is no pointer to check and no `_wrap` to do. The C symbols spell the C++ trailing underscore as `_inplace` — `add_` would be a legal C identifier but a confusing export. Python's `add_`/`add_out` return `self`/`out` so calls chain, and `__iadd__` and friends return `self` so `a += b` does not silently rebind the name to a new tensor.

### DLPack

DLPack is the tensor exchange standard of PyTorch, NumPy, CuPy and JAX: a C struct (`DLTensor`: data pointer, device, dtype, shape, strides, byte offset) wrapped in a `DLManagedTensor` whose **deleter** the consumer calls when done. In Python it travels as a PyCapsule named `"dltensor"`, renamed `"used_dltensor"` by whoever consumes it. OpenMat speaks it both ways, zero-copy: `torch.from_dlpack(t)` / `np.from_dlpack(t)` go through `Tensor.__dlpack__`, and `openmat.from_dlpack(x)` wraps a foreign tensor. The point is to call OpenMat's fused kernels on PyTorch tensors inside PyTorch code.

- [headers/dlpack.h](headers/dlpack.h) reproduces the ABI of DLPack v0.8's unversioned `DLManagedTensor` (Apache-2.0, dmlc/dlpack). Only the legacy capsule is produced — every consumer accepts it. One visible consequence: NumPy marks an array made from a legacy capsule **read-only**, since that format cannot say otherwise; write through the OpenMat side.
- **Export** (`om_tensor_<dtype>_to_dlpack`, [src/python/openmat_capi.cpp](src/python/openmat_capi.cpp) `DLPackExport`): the context holds an `alias()` of the tensor (so the storage outlives every OpenMat handle), the `int64_t` shape/stride arrays, and **one C-side reference on the storage's stream**, released in the deleter after the storage is freed. Without it, a PyTorch tensor outliving every OpenMat object — the `Stream` included — would free into a destroyed stream. The deleter touches only C++, so it is safe from any thread, with or without the GIL.
- **Import** (`om_tensor_<dtype>_from_dlpack` → `Tensor::from_external`): a `Storage` built over borrowed memory holds a `release` callback that runs the producer's deleter when the last view dies, instead of freeing through the allocator. Everything is validated *before* the tensor exists (dtype, device CPU/CUDA/CUDAHost/managed, rank ≤ 8, non-negative strides — OpenMat strides are `size_t`), so a refused import leaves the capsule unconsumed and its own destructor runs the deleter. A 0-d tensor imports as shape `{1}`.
- **Python** ([python/openmat/_dlpack.py](python/openmat/_dlpack.py)): capsules are made and read through `ctypes.pythonapi`. The capsule names are module-level `bytes` because `PyCapsule_New` stores the pointer, not a copy. The capsule destructor is a module-level `CFUNCTYPE` taking the capsule as `void*` — as a `py_object` it would take a reference on an object whose refcount is already 0 — and runs the deleter only if the capsule is still named `"dltensor"`.
- **Streams**: imports ask a CUDA producer for the legacy default stream (`__dlpack__(stream=1)`), which is where OpenMat's default-stream ops run, so the producer orders its pending work first. On export OpenMat has no events (roadmap P4), so unless both sides are on the legacy default stream (`stream` in `(None, 1)` and the tensor's storage on the default stream) it synchronizes the device before handing the capsule over; `stream=-1` skips that.
- Cost is per call, not per byte: 10.6 µs torch → OpenMat and 5.3 µs OpenMat → torch, the same at 1 K and 16 M elements (GB10, Release, 2026-09-29). Import is the dearer side because it goes through `__dlpack_device__`, `__dlpack__`, three capsule calls and `om_dlpack_info` in ctypes.
- Only `float32` and `int32` cross; anything else is a `TypeError` naming the DLPack code. [tests/test_dlpack.cpp](tests/test_dlpack.cpp) checks the ownership rules at the C-ABI (deleter once, after the last view; never for a refused import; export keeps its stream alive) and runs under memcheck in CI.

The Python package is `Tensor` + `Stream` + a `DType` registry ([python/openmat/_dtypes.py](python/openmat/_dtypes.py)); host tensors expose `__array_interface__` (zero-copy `np.asarray`), CUDA tensors `__cuda_array_interface__`. See [python/README.md](python/README.md) for the user-facing surface.

## Benchmarking

Five harnesses (`bench_vs`, `bench_rank_sweep`, `bench_omp`, `bench_inplace`, the `OPENMAT_HOST_CACHE_BYTES=0` A/B), all requiring a **Release** build — Debug numbers are meaningless. See the `benchmarking` skill for what each one is for and the two traps that have already produced wrong conclusions in the reports.

## Reference docs

- [README.md](README.md) — measured stream benchmarks (RTX 4060 + GB10) and the rationale behind each design decision.
- [benchmark_report.md](benchmark_report.md) — OpenMat vs NumPy vs PyTorch, with the root-cause analysis behind each gap.
- [stream_perf_report.md](stream_perf_report.md) — raw `test_stream_perf` output, plus `test_benchmarks` and `test_stress`. The **GB10 appendix is the current source** (re-measured 2026-09-04, medians of 9 runs); the RTX 4060 sections above it are a 2024 snapshot of a different code state and should not be quoted as current. Nine runs rather than four because the suite times a single un-warmed run per variant and the spread is wide — the sequential chain alone produced single runs from 0.98× to 1.16×.
- [docs/fused_operations.md](docs/fused_operations.md) — fusion design walkthrough.
- [docs/roadmap.md](docs/roadmap.md) — planned work with a done/not-done priority table at the end (written in Italian).
