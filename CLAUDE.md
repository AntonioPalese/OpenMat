# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

Requirements: NVIDIA GPU (compute capability ≥ 7.0), CUDA Toolkit ≥ 12.0, CMake ≥ 3.18, GCC ≥ 10.

```bash
# Full clean build (also regenerates compile_commands.json)
./compile.sh

# Or manually:
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make -j$(nproc)
```

The build produces:
- `build/OpenMat.so` — shared library
- `build/OpenMat_app` — main executable
- `build/tests/tensor_ops_test` — test binary

CUDA architecture is hardcoded to `sm_61` in [CMakeLists.txt](CMakeLists.txt):26. Change `CMAKE_CUDA_ARCHITECTURES` if targeting a different GPU.

## Tests

Uses GoogleTest (fetched automatically by CMake via FetchContent).

```bash
# Run all tests
cd build && ctest

# Run test binary directly (shows individual test output)
./build/tests/tensor_ops_test

# Run a single test by name
./build/tests/tensor_ops_test --gtest_filter="TensorArithmetic.CPUOperations"
```

## Architecture

Everything lives under the `om` namespace. The data flow for a tensor op is:

```
Tensor<T>::operator+()
  → _add(TensorView, TensorView, TensorView, DEVICE_TYPE)   [kernel_launcher.inl]
    → add_dispatch<DEVICE_TYPE::CPU/CUDA, T>::exec()        [kernel_launcher.h]
      → add_cpu() or launch_add()                           [ops/cpu/ or ops/kernels/]
        → rank-specialized CUDA kernel or flat CPU loop
```

**`Tensor<T>`** ([headers/tensor.cuh](headers/tensor.cuh), [headers/tensor.inl](headers/tensor.inl)) — owning N-dimensional tensor. Stores shape, row-major strides, a raw `T*`, a `Device`, and a `unique_ptr<Allocator<T>>`. Construction allocates memory via `AllocatorFactory`. Copy deep-copies via the allocator; move transfers ownership and nulls the source pointer.

**`Allocator<T>` / `AllocatorFactory<T>`** ([headers/allocator.h](headers/allocator.h)) — abstract base with two concrete implementations: `CpuAllocator` (malloc/free/memcpy) and `GpuAllocator` (cudaMalloc/cudaFree/cudaMemcpy). `AllocatorFactory::create(DEVICE_TYPE)` selects the right one at `Tensor` construction time.

**`Device`** ([headers/mat_utils.h](headers/mat_utils.h)) — lightweight struct (`m_Id`, `m_Str`, `m_Dt`). Constructed from a string like `"cpu:0"` or `"cuda:0"`.

**`TensorView<T>`** ([headers/tensor_view.cuh](headers/tensor_view.cuh)) — non-owning host-side view: raw pointer + shape/stride pointers + rank. `__host__`-only. Converted to `DeviceTensorView` via `.as_device_tw()` before kernel launch.

**`DeviceTensorView<T>`** ([headers/device_tensor_view.cuh](headers/device_tensor_view.cuh)) — non-owning device-side view. Its constructor allocates and uploads shape/stride arrays to GPU memory with `cudaMalloc`/`cudaMemcpy`; destructor frees them. Move-only. Operator `()` is `__device__`-only and uses `device_load` for reads.

**Kernel dispatch** ([headers/kernel_launcher.h](headers/kernel_launcher.h), [headers/kernel_launcher.inl](headers/kernel_launcher.inl)) — two macro families:
- `DEFINE_DEVICE_DISPATCH_BINARY_H(OP_NAME, CPU_FUNC, CUDA_FUNC)` — declares `op_dispatch<DEVICE_TYPE, T>` template structs that call `CPU_FUNC` or `CUDA_FUNC`.
- `DEFINE_DEVICE_DISPATCH_BINARY_INL(OP_NAME)` — defines the free function `_op(lhs, rhs, dst, DEVICE_TYPE)` that switches at runtime and calls the matching dispatch struct.
- Unary (scalar) variants follow the same pattern with `DEFINE_DEVICE_DISPATCH_UNARY_H/INL`.

**Rank-specialized CUDA kernels** ([headers/ops/kernels/binary_op_macros.cuh](headers/ops/kernels/binary_op_macros.cuh)) — `DEFINE_BINARY_OP_LAUNCH(OP_NAME)` generates a `launch_op` function that switches on `lhs.rank` (1–4) and selects a matching kernel with a rank-tuned grid/block layout. Rank ≥ 5 falls back to a flat 1D kernel (`_kernel_nd`) that reconstructs multi-indices from a linear index. Explicit template instantiations for `float`, `int`, `char`, `float16_t` are emitted by `DEFINE_BINARY_OP_LAUNCH_FRW_DEC`.

**Ops layout:**
```
headers/ops/cpu/        ← CPU op declarations (macro-generated inline functions)
src/ops/cpu/            ← CPU op .cpp translation units
headers/ops/kernels/    ← CUDA kernel declarations and launch macros (.cuh)
src/ops/kernels/        ← CUDA kernel .cu translation units
```

**Adding a new op:** define the kernel body expression in `src/ops/kernels/` using `DEFINE_BINARY_OP_KERNEL_K1/K2/K3/K4/ND` and `DEFINE_BINARY_OP_LAUNCH`, add the CPU implementation in `src/ops/cpu/`, declare both in their respective headers, then register the dispatch pair in [headers/kernel_launcher.h](headers/kernel_launcher.h) with `DEFINE_DEVICE_DISPATCH_BINARY_H` and in [headers/kernel_launcher.inl](headers/kernel_launcher.inl) with `DEFINE_DEVICE_DISPATCH_BINARY_INL`.

**Supported dtypes** (via `om::dtype<T>()` specializations): `float`, `double`, `int`, `char`, `float16_t`.
