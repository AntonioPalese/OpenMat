# OpenMat Repository Summary

## Overview

**OpenMat** is a from-scratch **CUDA tensor framework in C++17/CUDA 17**, with a
**ctypes-based Python package** on top. Every kernel is hand-written — there is no
cuBLAS/cuDNN dependency. It is built to explore GPU kernel authoring, allocator
architecture, stream semantics and Python FFI design from first principles.

Everything C++ lives under the `om` namespace.

---

## Project Structure

```
OpenMat/
├── headers/
│   ├── tensor.cuh / tensor.inl      # Core Tensor<T> template (owning N-D tensor)
│   ├── tensor_view.cuh              # Host-side non-owning view
│   ├── device_tensor_view.cuh       # Device-side view (inline shape/stride arrays)
│   ├── stream.h                     # om::Stream — move-only RAII cudaStream_t
│   ├── allocator.h / .inl           # Cpu/Gpu allocators + factory, async variants
│   ├── kernel_launcher.h / .inl     # Legacy macro dispatch (mostly superseded)
│   ├── mat_utils.h / .inl           # Device, DEVICE_TYPE, dtype<T>()
│   ├── ops/cpu/                     # binary/unary macros, matmul, reduce, transpose
│   ├── ops/kernels/                 # CUDA launch macros, fused ops, matmul,
│   │                                #   reduce, transpose, fill
│   └── type_traits/types.cuh        # float16_t wrapper, is_extended_arithmetic
├── src/
│   ├── main.cpp                     # OpenMat_app entry point
│   ├── ops/cpu/*.cpp                # CPU translation units
│   ├── ops/kernels/*.cu             # CUDA translation units
│   └── python/
│       ├── openmat_capi.cpp         # C-ABI boundary (compiled into OpenMat.so)
│       └── openmat_capi_impl.inc    # Per-dtype body, included twice (float, int)
├── python/
│   ├── openmat/                     # Tensor, Stream, DType registry, _clib loader
│   ├── tests/                       # pytest suites
│   ├── hatch_build.py               # Copies OpenMat.so into the wheel
│   └── pyproject.toml
├── tests/                           # 12 GoogleTest suites, one binary each
├── docs/                            # fused_operations.md, roadmap.md (Italian)
├── CMakeLists.txt, compile.sh
└── README.md, CLAUDE.md, stream_perf_report.md
```

---

## Core Components

### 1. `Tensor<T>` (`headers/tensor.cuh`, `tensor.inl`)

Owning N-D tensor: shape, row-major strides, raw `T*`, a `Device`, an
`om::Stream m_Stream`, and a `unique_ptr<Allocator<T>>`. Copy deep-copies; move
transfers ownership and nulls the source.

Surface:
- **Factories**: `zeros`, `ones`, `full`, `from_vector` (plus a stream overload)
- **Binary ops**: `add`/`sub`/`mul`/`div` + `+ - * /`, tensor–tensor and tensor–scalar
- **`matmul`** — 2D only, both backends, no batching or broadcasting
- **Reductions**: `sum`, `mean`, `min`, `max` — synchronous, return a host scalar
- **Shape**: `reshape`, `flatten`, `squeeze`, `unsqueeze` — **deep copies, not views**
- **`transpose()`** (rank-2 only, throws otherwise) and **`permute(axes)`**
- **Fused**: `apply`, `apply_binary`, `relu`, `sigmoid`, `scale_shift`, `shift_scale`,
  `fused_add_mul`, `fused_sub_mul`, `fused_mul_add`, `fused_div_add`
- **Transfer**: `to(device)`, `cpu()`, `cuda()`, `copyToHost`, `copyToDevice`
- **Metadata**: `shape`, `stride`, `size`, `rank`, `device`, `device_type`, `dtype`, `stream`

```cpp
om::Tensor<float> a({2, 3}, om::Device(0, om::DEVICE_TYPE::CUDA));
a.fill(1.0f);
auto b = a + a;              // synchronous
om::Stream s;
auto c = a.add(a, s);        // enqueued on s
s.synchronize();
```

### 2. Streams are the canonical execution path

Every op exists in two forms and **the stream form is the real implementation**;
the no-stream form is a one-line delegate passing `Stream::default_stream()` — a
non-owning wrapper around `nullptr`. This is why there is no duplicated dispatch code.

**Stream-ownership invariant:** `cudaMallocAsync` memory belongs to a stream-ordered
pool and must be freed on the stream that allocated it. Each `Tensor` therefore stores
`m_Stream`, and a private `Tensor(shape, device, Stream)` constructor lets stream
overloads allocate results on the enqueuing stream.

### 3. Allocators (`headers/allocator.h`, `.inl`)

- **`CpuAllocator<T>`** — malloc/free/memcpy
- **`GpuAllocator<T>`** — cudaMalloc/cudaFree/cudaMemcpy, with
  `cudaMallocAsync`/`cudaFreeAsync` under `CUDART_VERSION >= 11020`
- **`AllocatorFactory<T>`** — picks one from `DEVICE_TYPE`

The base class declares `allocate_async`, `deallocate_async`, `copy_async`,
`copy_host_to_device_async`, `copy_device_to_host_async` with **synchronous default
implementations**, so a subclass overrides only what it can genuinely do async.

### 4. Views

- **`TensorView<T>`** — host-only, non-owning (pointer + shape/stride pointers + rank).
- **`DeviceTensorView<T>`** — device-side, with **fixed inline arrays**
  (`size_t shape[MAX_RANK]`, `MAX_RANK = 8`) so the struct is trivially copyable and
  passed **by value** in the kernel parameter block. Deliberately no device allocation
  and no pointer members — an earlier design cudaMalloc'd shape/stride per view
  (6 allocations per binary op). Rank > 8 trips an assert.

### 5. Dispatch: two paths, one mostly dead

`kernel_launcher.h`/`.inl` still hold the macro-generated `_dispatch` structs, but
since the stream refactor `tensor.inl` branches on `device_type()` itself and calls
`add_cpu(...)` / `launch_add(..., s.get())` directly — the dispatch structs have no
`cudaStream_t` parameter. Today only `_fill` still goes through them. Treat the
dispatch macros as **legacy**.

Effective data flow for `a + b`:

```
Tensor<T>::operator+()
  → Tensor<T>::add(rhs, Stream::default_stream())      [tensor.inl]
    → add_cpu(...)  or  launch_add(..., stream)        [ops/cpu/ or ops/kernels/]
      → flat CPU loop  or  rank-specialized CUDA kernel
```

### 6. Rank-specialized kernels

`DEFINE_BINARY_OP_LAUNCH(OP)` generates `launch_OP(lhs, rhs, dst, cudaStream_t)`,
switching on rank and picking a tuned grid/block layout (1D `dim3(16)`,
2D `dim3(16,16)`, 3D/4D `dim3(8,8,8)`). Rank ≥ 5 falls back to a flat `OP_kernel_nd`
that reconstructs multi-indices from a linear index.

### 7. Fused operations (`headers/ops/kernels/fused_op.cuh`)

Functor-based fusion, no intermediate allocation: `Add`, `Mul`, `Div`, `Pow`, `ReLU`,
`Sigmoid`, `Compose<F,G>`, `BinaryAdd/Sub/Mul/Div`, `BinaryCompose`, driven by
`launch_apply_op` / `launch_apply_binary_op` (rank 1–4 kernels plus an `_nd` fallback).
Both `apply` and `apply_binary` have real CPU branches.

⚠️ **Explicit instantiations** in `src/ops/kernels/fused_op.cu` must list every
`(T, Op)` pair used from a `.cpp` TU — a missing one is a link error.

### 8. Reductions

GPU: two-phase shared-memory tree reduction plus warp shuffle (`__shfl_down_sync`).
CPU: plain loops in `headers/ops/cpu/reduce_cpu.h`.

### 9. Dtypes

`om::dtype<T>()` covers `float`, `double`, `int`, `char`, `float16_t`. Kernel
instantiations cover `float`, `int`, `char`, `float16_t` — **`double` has no GPU
instantiation**.

`float16_t` is a hand-rolled `__half` wrapper with `__host__ __device__` conversions
and free operators using `__hadd`/`__hsub`/… on `__CUDA_ARCH__ >= 530`, falling back
to `float` math otherwise. Generic code must gate on `is_extended_arithmetic<T>`, not
`std::is_arithmetic`, or half precision is rejected.

---

## Python Package

A **ctypes** binding (not pybind) over the C-ABI in `src/python/openmat_capi.cpp`.

```bash
cd python && uv sync && uv pip install -e .
```

`python/openmat/_clib.py` locates the library as `$OPENMAT_LIB` → bundled
`openmat/OpenMat.so` → `<repo>/build/OpenMat.so`, so the bindings work straight from a
source checkout after `./compile.sh`.

**Surface** (`openmat`): `Tensor`, `Stream`, `DType`, `dtype`, `float32`, `int32`,
`cuda_is_available`, `device_count`, `synchronize`, plus module-level factory aliases
`zeros`, `ones`, `full`, `empty`, `arange`, `from_list`, `from_numpy`.

`Tensor` mirrors `om::Tensor<T>`: factories, arithmetic and operators (incl. `__matmul__`,
reflected ops, `__neg__`), reductions, `reshape`/`flatten`/`squeeze`/`unsqueeze`,
`transpose`/`T`/`permute`, the fused ops, `cpu`/`cuda`/`to`/`astype`, indexing
(`__getitem__`/`__setitem__`), `tolist`, `item`, `numpy`. Host tensors expose
`__array_interface__` (zero-copy `np.asarray`); CUDA tensors expose
`__cuda_array_interface__`. Ops with a C++ stream overload take a `stream=None` kwarg.

**C-ABI conventions:**
- Tensor handles are opaque `void*`; every `_create`/`_copy` needs exactly one `_destroy`.
- Pointer-returning functions return `nullptr` on failure, int-returning ones non-zero;
  both write the message into a caller-supplied `char* errbuf` (Python passes 512 bytes).
  No exception crosses the boundary (`OM_GUARD_PTR` / `OM_GUARD_INT` / `OM_GUARD_VAL`).
- Infallible metadata getters take no errbuf.
- The per-dtype surface is **one body included twice**: `openmat_capi_impl.inc` is
  included with `OM_T=float, OM_SFX=float` and `OM_T=int, OM_SFX=int`, so
  `om_tensor_float_add` never greps as a literal definition. Adding a dtype is one
  `#define`/`#include`/`#undef` block — provided kernels are instantiated for it.
- Dtype-independent runtime API: `om_cuda_device_count`, `om_cuda_is_available`,
  `om_device_synchronize`, `om_stream_create/retain/release/destroy/synchronize/handle`.

**Streams are reference-counted at the C boundary** (`StreamBox`), not in Python:
Python's cyclic GC finalizes in arbitrary order, so a `Stream` could be torn down before
tensors that still own pool memory from it. Each `Tensor` holds an integer handle plus
one C-side reference. `Stream.close()` is therefore safe while its tensors are alive.

Adding a Python-visible method = three edits: the `.inc`, the `ctypes` signatures in
`_declare_dtype()`, and the wrapper in `python/openmat/tensor.py`.

---

## Build System

CMake ≥ 3.24 (for `CMAKE_CUDA_ARCHITECTURES=native`), CUDA Toolkit ≥ 11.2 (for
`cudaMallocAsync`), C++17/CUDA 17. Verified on CUDA 13.0 / GCC 13.3 / CMake 3.28 /
GB10 (sm_121), all 12 suites passing.

```bash
./compile.sh          # clean rebuild + refreshes compile_commands.json
# or
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .. && make -j$(nproc)
```

Outputs `build/OpenMat.so` (also what the Python package loads), `build/OpenMat_app`,
and `build/tests/test_*`.

Gotchas:
- **`CMAKE_LIBRARY_PATH` must be set in the environment** (colon-separated); its entries
  become `target_link_directories` for `-lcuda`/`-lcudart`. CMake only *warns* when it is
  unset — `OpenMat.so` still builds, then every executable fails with `cannot find -lcudart`.
  ```bash
  export CMAKE_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/$(uname -m)-linux-gnu"
  ```
- The `CMAKE_CUDA_ARCHITECTURES` guard sits **above `project()`** on purpose — `project(…
  LANGUAGES CUDA)` defines the variable itself, so a later `if(NOT DEFINED …)` never fires.
- Default `CMAKE_BUILD_TYPE` is `Debug`; pass `-DCMAKE_BUILD_TYPE=Release` before benchmarking.
- Sources use `file(GLOB …)` — a new `.cpp`/`.cu` needs a CMake re-run, not just `make`.
- `cmake/detect_cuda_arch.cmake` and `scripts/detect_archs.py` are **stale and unused**
  (`nvcc --list-gpus` was removed in CUDA 13); `native` replaces both.
- The root `.gitignore` starts with `*build*`, which matches `python/hatch_build.py` —
  kept only by an explicit `!python/hatch_build.py`. Beware new files with "build" in the name.

---

## Testing

**C++ — GoogleTest via FetchContent, one binary per suite** (`add_om_test` in
`tests/CMakeLists.txt`):

```bash
cd build && ctest                     # all suites
./build/tests/test_arithmetic         # one suite
./build/tests/test_arithmetic --gtest_filter="TensorArithmetic.CPUOperations"
```

| Suite | Focus |
|-------|-------|
| `test_arithmetic` | binary/scalar ops, CPU + GPU |
| `test_fused_ops` | fused functors, both backends |
| `test_device_transfer` | `to`/`cpu`/`cuda`, sync + async |
| `test_factory` | `zeros`/`ones`/`full`/`from_vector` |
| `test_reductions` | `sum`/`mean`/`min`/`max` |
| `test_reshape` | reshape/flatten/squeeze/unsqueeze |
| `test_transpose` | transpose + permute |
| `test_streams` | stream semantics |
| `test_allocator_stream` | stream-ordered alloc/free invariant |
| `test_benchmarks`, `test_stress`, `test_stream_perf` | timing/soak — **not** correctness; meaningless in Debug |

**Python — pytest** (`python/tests/`): `test_tensor.py`, `test_tensor_api.py`,
`test_dtypes.py`, `test_streams.py`. `conftest.py` provides a `device` fixture running
each test on both backends plus a `requires_cuda` marker.

```bash
OPENMAT_LIB=build/OpenMat.so python python/test_bindings.py   # smoke script
cd python && pytest
```

---

## Key Design Patterns

1. **Stream-first single source of truth** — sync API delegates to the async one
2. **RAII everywhere** — `unique_ptr<Allocator>`, move-only `Stream`
3. **Trivially-copyable device metadata** — no per-launch device allocation
4. **Factory pattern** — `AllocatorFactory` for device-agnostic allocation
5. **Macro code generation** — kernels, launches, CPU ops, and the per-dtype C-ABI
6. **Functor composition** — fusion without intermediate buffers
7. **Reference counting at the FFI boundary** — not left to Python's GC

---

## Development Status

| Feature | Status |
|---------|--------|
| Binary + scalar ops (add, sub, mul, div), CPU & CUDA | ✅ |
| Memory allocator abstraction (sync + async) | ✅ |
| CUDA streams across the whole tensor surface | ✅ |
| Device transfer (`to`/`cpu`/`cuda`, sync + async) | ✅ |
| Factories (`zeros`/`ones`/`full`/`from_vector`) | ✅ |
| Reductions (`sum`/`mean`/`min`/`max`) | ✅ |
| Shape ops (reshape/flatten/squeeze/unsqueeze) | ✅ (deep copies, not views) |
| `transpose` / `permute` | ✅ (transpose is rank-2 only) |
| `matmul` | ✅ 2D only — no batching, no broadcasting |
| Fused ops incl. CPU path | ✅ |
| C++ test suites (12) | ✅ |
| Python bindings (ctypes, float32 + int32) | ✅ |
| Python test suites (4) | ✅ |
| cuBLAS integration | 🔄 Planned (roadmap 5.2) |
| Random init (cuRAND) | 🔄 Planned (roadmap 3.3) |
| Autograd | ❌ Not planned in current roadmap |

Known limitations: no views/aliasing anywhere, no broadcasting, `double` has no GPU
kernels, `Tensor<double>`/`Tensor<char>` are not exported to Python, and rank is capped
at 8 on the device side.

---

## Technology Stack

- **Languages**: C++17, CUDA C++17, Python 3
- **Build**: CMake (+ `compile.sh`), hatch/uv for the Python package
- **Testing**: GoogleTest (C++), pytest (Python)
- **GPU**: NVIDIA CUDA Toolkit ≥ 11.2
- **Platform**: Linux

---

## Reference Docs

- [README.md](README.md) — design rationale and stream benchmarks
- [CLAUDE.md](CLAUDE.md) — working notes and build gotchas for contributors
- [stream_perf_report.md](stream_perf_report.md) — raw `test_stream_perf` output
- [docs/fused_operations.md](docs/fused_operations.md) — fusion design walkthrough
- [docs/roadmap.md](docs/roadmap.md) — planned work + done/not-done table (Italian)
- [python/README.md](python/README.md) — user-facing Python surface
