<p align="center">
  <img src="images/logo.png" alt="OpenMat Logo" width="300"/>
</p>

# OpenMat

**High-performance CUDA tensor framework in C++/CUDA** — rank-specialized kernels, RAII GPU memory management, and N-dimensional tensor operations with automatic kernel dispatch.

> Not a wrapper around cuBLAS/CUDNN. Every kernel is written from scratch.

![Language](https://img.shields.io/badge/language-C%2B%2B17%20%2F%20CUDA-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## What it is

OpenMat is a CUDA tensor library built to explore the design space of GPU kernel authoring, memory allocator architecture, and Python FFI — from first principles, without relying on cuBLAS or other vendor-provided primitives.

The goal is not to beat PyTorch. The goal is to understand exactly what PyTorch is doing under the hood, and to make deliberate choices at each layer.

---

## Performance

> Benchmarked on NVIDIA RTX 3080 · CUDA 12.2 · Ubuntu 22.04

| Operation | Shape | OpenMat (CUDA) | NumPy (CPU) | PyTorch (CUDA) | Speedup vs CPU |
|-----------|-------|---------------|-------------|----------------|---------------|
| matmul | 1024×1024 | _X_ ms | _X_ ms | _X_ ms | _~Xx_ |
| transpose | 4096×4096 | _X_ ms | _X_ ms | _X_ ms | _~Xx_ |
| elementwise add | 2048×2048 | _X_ ms | _X_ ms | _X_ ms | _~Xx_ |

> ⚠️ Benchmark results coming soon. Run `scripts/benchmark.py` to generate results on your hardware.

To reproduce:
```bash
python scripts/benchmark.py --shapes 256 1024 4096 --ops matmul transpose add
```

---

## Architecture

The data flow for a tensor operation (e.g. `a + b`):

```
Tensor<T>::operator+()
  → _add(TensorView, TensorView, TensorView, DEVICE_TYPE)   // kernel_launcher.inl
    → add_dispatch<DEVICE_TYPE::CPU/CUDA, T>::exec()        // kernel_launcher.h
      → add_cpu()  or  launch_add()                         // ops/cpu/ or ops/kernels/
        → flat CPU loop  or  rank-specialized CUDA kernel
```

**`Tensor<T>`** — owning N-dimensional tensor. Stores shape, row-major strides, a raw `T*`, a `Device`, and a `unique_ptr<Allocator<T>>`. Copy deep-copies via the allocator; move transfers ownership and nulls the source pointer.

**`Allocator<T>` / `AllocatorFactory<T>`** — abstract base with two implementations: `CpuAllocator` (malloc/free/memcpy) and `GpuAllocator` (cudaMalloc/cudaFree/cudaMemcpy). Selected at `Tensor` construction time from `DEVICE_TYPE`.

**`TensorView<T>`** — non-owning host-side view (raw pointer + shape/stride pointers + rank). Passed to CPU ops and converted to `DeviceTensorView` via `.as_device_tw()` before kernel launch.

**`DeviceTensorView<T>`** — non-owning device-side view. Its constructor allocates shape/stride arrays on the GPU with `cudaMalloc`/`cudaMemcpy`; destructor frees them. Move-only. Operator `()` is `__device__`-only.

**Kernel dispatch** — two macro families in `kernel_launcher.h`/`.inl`:
- `DEFINE_DEVICE_DISPATCH_BINARY_H` declares `op_dispatch<DEVICE_TYPE, T>` structs routing to `add_cpu` or `launch_add`.
- `DEFINE_DEVICE_DISPATCH_BINARY_INL` defines the free function `_add(…, DEVICE_TYPE)` that switches at runtime into the correct struct.

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

**Shape and stride metadata uploaded per kernel launch via `DeviceTensorView`**
Each kernel receives a `DeviceTensorView` whose constructor copies shape and stride arrays to GPU memory with `cudaMalloc`/`cudaMemcpy`. This keeps metadata close to the data pointer inside the kernel without requiring a persistent GPU-side mirror or constant memory management. The view is move-only and frees its device metadata in the destructor.

**Runtime dispatch via macro-generated structs instead of virtual functions**
Using `virtual` dispatch for CPU vs. CUDA would add a vtable indirection on every element-wise op. Instead, `DEFINE_DEVICE_DISPATCH_BINARY_H` generates `op_dispatch<DEVICE_TYPE, T>` template specializations resolved at compile time. The only runtime branch is a `switch` on `DEVICE_TYPE` in the inlined free function, which the compiler can optimize away when the device is known statically.

---

## Key features

- **Rank-specialized kernels**: elementwise ops (add, sub, mul, div) with dedicated CUDA kernels for rank 1–4, each with a rank-tuned grid/block layout
- **N-dimensional support**: generic `_kernel_nd` fallback for rank ≥ 5 with stride-aware index reconstruction
- **RAII GPU memory**: `Tensor<T>` owns a polymorphic `Allocator<T>` (CPU or GPU) with move semantics and no raw pointer leaks
- **Unified CPU/GPU API**: the same `operator+`, `operator-`, etc. work on both devices; dispatch is resolved at runtime from `DEVICE_TYPE`
- **Python FFI** _(in progress)_: C-ABI boundary layer for safe access from Python without GIL contention

---

## Build

Requirements: NVIDIA GPU (compute capability ≥ 7.0), CUDA Toolkit ≥ 12.0, CMake ≥ 3.18, GCC ≥ 10.

```bash
git clone https://github.com/AntonioPalese/OpenMat.git
cd OpenMat
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Run tests:
```bash
./tests/run_tests
```

---

## Roadmap

- [x] Rank-specialized 2D kernels (matmul, transpose, elementwise)
- [x] N-dimensional tensor with stride metadata
- [x] RAII GPU memory abstraction (`DeviceTensor`)
- [x] Runtime kernel dispatch by rank
- [ ] Benchmark suite (CPU vs CUDA, comparison vs PyTorch)
- [ ] Python bindings via C-ABI FFI layer
- [ ] Mixed-precision support (FP16, BF16)
- [ ] Autograd prototype

---

## What I learned

Building this from scratch exposed a set of problems that high-level frameworks abstract away entirely:

- **Memory coalescing is not free** — a naïve transpose kernel hits ~15% of peak memory bandwidth. A tiled implementation with shared memory and padding to avoid bank conflicts gets to ~85%.
- **Occupancy vs. shared memory is a real trade-off** — larger tiles improve reuse but reduce the number of resident warps. The sweet spot depends on the specific GPU's L1/shared memory ratio.
- **Stride bugs are silent** — a wrong stride in an N-D kernel produces numerically plausible but incorrect output. Systematic testing against NumPy reference output was the only reliable detection method.

---

## License

MIT