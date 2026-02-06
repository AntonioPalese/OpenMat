# OpenMat Repository Summary

## Overview

**OpenMat** is a didactic open-source framework for **GPU-accelerated matrix/tensor operations** using NVIDIA CUDA. It's designed as both a learning tool and a performance benchmark for understanding low-level C++/CUDA programming.

---

## Project Structure

```
OpenMat/
├── headers/                  # Header files (.h, .cuh, .inl)
│   ├── tensor.cuh           # Core Tensor class template
│   ├── tensor_view.cuh      # Lightweight view into tensor data
│   ├── device_tensor_view.cuh # Device-side tensor view
│   ├── allocator.h          # Memory allocator abstraction
│   ├── kernel_launcher.h    # Dispatches ops to CPU/GPU
│   ├── mat_utils.h          # Device enum, utilities
│   ├── ops/
│   │   ├── cpu/             # CPU operation headers
│   │   └── kernels/         # CUDA kernel headers
│   └── type_traits/         # Type trait utilities
├── src/                      # Source files
│   ├── main.cpp             # Application entry point
│   ├── mat_utils.cpp        # Utility implementations
│   ├── ops/
│   │   ├── cpu/             # CPU operation implementations
│   │   │   ├── binary_ops.cpp
│   │   │   ├── unary_ops.cpp
│   │   │   └── fill_cpu.cpp
│   │   └── kernels/         # CUDA kernel implementations
│   │       ├── binary_ops.cu
│   │       ├── unary_ops.cu
│   │       └── fill_gpu.cu
│   └── type_traits/
├── tests/                    # Unit tests (Google Test)
│   └── tensor_ops_test.cpp
├── cmake/                    # CMake modules
├── CMakeLists.txt           # Build configuration
└── README.md
```

---

## Core Components

### 1. Tensor Class (`headers/tensor.cuh`)

The central data structure supporting:
- **N-dimensional tensors** with arbitrary shape
- **CPU and CUDA device** placement (via `Device` abstraction)
- **Operator overloading**: `+`, `-`, `*`, `/` for element-wise operations
- **Scalar operations**: tensor-scalar arithmetic
- **Memory management**: automatic via custom allocators
- **Views**: lightweight `TensorView` for kernel passing

```cpp
Tensor<float> a({2, 3}, Device(0, DEVICE_TYPE::CUDA));
a.fill(1.0f);
Tensor<float> b = a + a;  // GPU-accelerated addition
```

### 2. Allocator Abstraction (`headers/allocator.h`)

Polymorphic memory management with:
- **`CpuAllocator<T>`**: Uses `malloc`/`free` + `memcpy`
- **`GpuAllocator<T>`**: Uses `cudaMalloc`/`cudaFree` + `cudaMemcpy`
- **`AllocatorFactory<T>`**: Creates allocator based on `DEVICE_TYPE`

### 3. Kernel Launcher (`headers/kernel_launcher.h`)

Dispatch system using C++ template specialization:
- **Compile-time dispatch** between CPU and CUDA implementations
- **Macro-based generation** of dispatch structs for operations

Supported operations:
| Operation | Binary (Tensor-Tensor) | Unary (Tensor-Scalar) |
|-----------|------------------------|----------------------|
| Add       | ✅ `add`, `+`          | ✅ `add_k`           |
| Subtract  | ✅ `sub`, `-`          | ✅ `sub_k`           |
| Multiply  | ✅ `mul`, `*`          | ✅ `mul_k`           |
| Divide    | ✅ `div`, `/`          | ✅ `div_k`           |
| Fill      | -                      | ✅ `fill`            |

### 4. CUDA Kernels (`src/ops/kernels/`)

- **`binary_ops.cu`**: Element-wise add, sub, mul, div kernels
- **`unary_ops.cu`**: Scalar arithmetic kernels
- **`fill_gpu.cu`**: Fill tensor with constant value

---

## Build System

- **CMake 3.22.1+** with CUDA language enabled
- **C++17 / CUDA 17** standard
- **Outputs**:
  - `OpenMat.so` - Shared library
  - `OpenMat_app` - Test executable
- **Target architecture**: SM 61 (Pascal, GTX 1000 series)

### Build Commands
```bash
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make
```

---

## Testing

Uses **Google Test** framework:
- `TensorArithmetic.CPUOperations` - CPU tensor arithmetic
- `TensorArithmetic.GPUOperations` - GPU tensor arithmetic with host verification

```bash
cd build && ctest
```

---

## Key Design Patterns

1. **Template Metaprogramming**: Device dispatch at compile time
2. **RAII**: Memory safety via `unique_ptr<Allocator>`
3. **View Pattern**: `TensorView` for efficient kernel arguments
4. **Factory Pattern**: `AllocatorFactory` for device-agnostic allocation
5. **Macro Code Generation**: Reduces boilerplate for operations

---

## Development Status

| Feature | Status |
|---------|--------|
| Basic tensor ops (add, mul, sub, div) | ✅ Complete |
| CPU + CUDA support | ✅ Complete |
| Memory allocator abstraction | ✅ Complete |
| Unit tests | ✅ Basic coverage |
| Python bindings | 🔄 Planned |
| Transpose, matmul | 🔄 Planned |
| Benchmark suite | 🔄 Planned |

---

## Technology Stack

- **Languages**: C++17, CUDA C++
- **Build**: CMake
- **Testing**: Google Test
- **GPU**: NVIDIA CUDA (requires CUDA Toolkit)
- **Platform**: Developed on Linux, Windows compatible
