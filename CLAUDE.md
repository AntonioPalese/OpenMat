# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Requirements: NVIDIA GPU, CUDA Toolkit ≥ 11.2 (for `cudaMallocAsync`), CMake ≥ 3.24 (for `CMAKE_CUDA_ARCHITECTURES=native`), C++17/CUDA 17 compiler. Verified building and passing all 12 suites on CUDA 13.0 / GCC 13.3 / CMake 3.28 / GB10 (sm_121).

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

Suites: `test_arithmetic`, `test_fused_ops`, `test_device_transfer`, `test_factory`, `test_reductions`, `test_benchmarks`, `test_reshape`, `test_transpose`, `test_streams`, `test_allocator_stream`, `test_stress`, `test_stream_perf`.

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
- **gpu** — a self-hosted runner labelled `self-hosted,linux,gpu`, needing nvcc and CMake ≥ 3.24 on PATH. Release build at `native` arch, the ten correctness suites, the Python suite, then `compute-sanitizer --tool memcheck --leak-check full` over `test_stress`, `test_allocator_stream` and `test_streams` (~30 s total). memcheck is the only thing that reports a stream-ownership violation near the call site instead of as an illegal access in an unrelated kernel later on.

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

**`Allocator<T>` / `AllocatorFactory<T>`** ([headers/allocator.h](headers/allocator.h), [.inl](headers/allocator.inl)) — `CpuAllocator` (malloc/free/memcpy) and `GpuAllocator` (cudaMalloc/cudaFree/cudaMemcpy). The base class declares `allocate_async`, `deallocate_async`, `copy_async`, `copy_host_to_device_async`, `copy_device_to_host_async` with **synchronous default implementations**, so a subclass overrides only what it can actually do async. `GpuAllocator` uses `cudaMallocAsync`/`cudaFreeAsync` under `#if CUDART_VERSION >= 11020`.

**Stream-ownership invariant:** `cudaMallocAsync` memory belongs to a stream-ordered pool and must be freed on the stream it was allocated on. That is why each `Tensor` stores `m_Stream` and the destructor calls `deallocate_async(m_Data, m_Stream.get())`. Breaking this shows up as an illegal memory access far from the real call site.

**`TensorView<T>`** ([headers/tensor_view.cuh](headers/tensor_view.cuh)) — non-owning host-side view (pointer + shape/stride pointers + rank), `__host__`-only. Converted with `.as_device_tw()` at launch.

**`DeviceTensorView<T>`** ([headers/device_tensor_view.cuh](headers/device_tensor_view.cuh)) — non-owning device-side view. Shape and stride are **fixed inline arrays** (`size_t shape[MAX_RANK]`, `MAX_RANK = 8`) filled from host at construction, so the struct is trivially copyable and passed **by value** into the kernel parameter block. There is deliberately no device allocation here — an earlier design cudaMalloc'd shape/stride per view (6 allocations per binary op). Do not reintroduce pointer members: raw arrays passed as kernel arguments decay to host pointers on the device side. Rank > 8 trips the constructor `assert`.

**`Device`** ([headers/mat_utils.h](headers/mat_utils.h)) — `m_Id`, `m_Str`, `m_Dt`. Constructible from `(id, DEVICE_TYPE)` or a string like `"cuda:0"`.

### Rank-specialized CUDA kernels

[headers/ops/kernels/binary_op_macros.cuh](headers/ops/kernels/binary_op_macros.cuh): `DEFINE_BINARY_OP_LAUNCH(OP)` generates `launch_OP(lhs, rhs, dst, cudaStream_t)` which switches on `lhs.rank` and picks a kernel with a rank-tuned grid/block layout (1D `dim3(16)`, 2D `dim3(16,16)`, 3D/4D `dim3(8,8,8)`). Rank ≥ 5 falls back to `OP_kernel_nd`, a flat 1D kernel reconstructing multi-indices from a linear index. `DEFINE_BINARY_OP_LAUNCH_FRW_DEC(OP)` emits explicit instantiations for `float`, `int`, `char`, `float16_t`.

**Ops layout:**
```
headers/ops/cpu/        ← CPU op declarations (macro-generated inline functions)
src/ops/cpu/            ← CPU op .cpp translation units
headers/ops/kernels/    ← CUDA kernel declarations and launch macros (.cuh)
src/ops/kernels/        ← CUDA kernel .cu translation units
```

**Adding a new binary op:**
1. `src/ops/kernels/binary_ops.cu` — kernel bodies via `DEFINE_BINARY_OP_KERNEL_K1/K2/K3/K4/ND` + `DEFINE_BINARY_OP_LAUNCH` + `DEFINE_BINARY_OP_LAUNCH_FRW_DEC`.
2. `headers/ops/kernels/binary_op_macros.cuh` — `DEFINE_BINARY_OP_LAUNCH_H` / `DEFINE_BINARY_OP_KERNEL_H`.
3. `src/ops/cpu/binary_ops.cpp` + `headers/ops/cpu/binary_op_macros.h` — CPU side.
4. `headers/tensor.cuh` / `.inl` — the `(rhs, const Stream&)` method plus the one-line no-stream delegate.
5. Only if a stream-less free function is wanted: register in `kernel_launcher.h`/`.inl`.

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

## Shape ops, transpose, matmul

These live outside the binary-op macro machinery and each has its own constraints.

**`reshape` / `flatten` / `squeeze` / `unsqueeze`** ([headers/tensor.inl](headers/tensor.inl)) are **deep copies, not views** — `reshape` does `Tensor out(*this)` (the copy ctor allocates and copies) and then rewrites `m_Shape`/`m_Stride` in place. Nothing in the library returns an aliasing view of another tensor's buffer, so unlike NumPy/PyTorch a reshape never lets a write show up through the original. The other three delegate to `reshape`. All are host-side only — no kernel, no stream overload — and `squeeze` of the last remaining axis yields shape `{1}` rather than a scalar.

**`transpose()` is rank-2 only** and throws otherwise; use `permute(axes)` for higher ranks. Both have real CPU and GPU paths ([headers/ops/cpu/transpose_cpu.h](headers/ops/cpu/transpose_cpu.h), [src/ops/kernels/transpose_gpu.cu](src/ops/kernels/transpose_gpu.cu)) and both have `(…, const Stream&)` overloads. `permute` validates axes on the host (length == rank, in range, no duplicates) before dispatching.

`launch_permute` takes the axes as a **host** `const size_t*` and copies them into an `AxesBuf` — a trivially-copyable struct passed **by value** into the kernel parameter block. This is the same rule as `DeviceTensorView`: no device allocation for small per-launch metadata, and no raw pointer members, which would decay to unusable host pointers on the device side.

**`matmul` is 2D-only** in both backends — rank != 2 or mismatched inner dimensions throw from `Tensor::matmul` ([headers/tensor.inl](headers/tensor.inl)) and again inside `matmul_cpu`. It is registered in the legacy dispatch table (`DEFINE_DEVICE_DISPATCH_BINARY_H(matmul, …)`) but `Tensor::matmul` bypasses it and calls `matmul_cpu` / `launch_matmul` directly, like every other stream-aware op. No batching, no broadcasting.

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

## Reference docs

- [README.md](README.md) — measured stream benchmarks (RTX 4060) and the rationale behind each design decision.
- [stream_perf_report.md](stream_perf_report.md) — raw `test_stream_perf` output.
- [docs/fused_operations.md](docs/fused_operations.md) — fusion design walkthrough.
- [docs/roadmap.md](docs/roadmap.md) — planned work with a done/not-done priority table at the end (written in Italian).
