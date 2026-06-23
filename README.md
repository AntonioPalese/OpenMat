# OpenMat

**High-performance CUDA tensor framework in C++/CUDA** — rank-specialized kernels, RAII GPU memory management, and N-dimensional tensor operations with automatic kernel dispatch.

> Not a wrapper around cuBLAS. Every kernel is written from scratch.

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

```
OpenMat
├── Tensor<T, Rank>          # Rank-specialized tensor: compile-time rank, runtime shape
├── DeviceTensor<T>          # RAII wrapper: cudaMalloc/cudaFree lifecycle
├── KernelDispatcher         # Runtime dispatch: selects optimal kernel by rank + shape
└── kernels/
    ├── matmul_rank2.cu      # Tiled shared-memory matmul for 2D tensors
    ├── matmul_nd.cu         # Generalized batched matmul for N-D tensors
    ├── transpose.cu         # Coalesced memory access transpose
    └── elementwise.cu       # Vectorized elementwise ops (add, mul, relu)
```

---

## Design decisions

These are the non-obvious choices made during development, and why.

**Rank-specialized kernels over a single generic kernel**
A generic N-D kernel requires runtime stride computation and can't use compile-time loop unrolling. For rank-2 (the common case), a specialized kernel using shared memory tiling and compile-time tile size achieves significantly better occupancy. The dispatcher selects the rank-2 path when possible and falls back to the N-D path otherwise.

**RAII for GPU memory via `DeviceTensor`**
Raw `cudaMalloc`/`cudaFree` pairs are error-prone in the presence of exceptions and early returns. `DeviceTensor` wraps the allocation in a move-only class with destructor-managed cleanup, making GPU memory lifetime deterministic and explicit — the same pattern used in PyTorch's `at::DataPtr`.

**Shape and stride metadata on-device**
Rather than maintaining a CPU-side mirror of shape/stride and passing it to every kernel launch, OpenMat stores a compact metadata struct in GPU constant memory. This avoids per-launch host-device copies for shape-dependent index computations.

**Runtime dispatch over template specialization for all ranks**
Full template specialization over rank (1–8) would produce a binary explosion and make Python FFI impractical. Instead, the dispatcher uses a rank enum at the boundary and calls the correct specialized kernel internally. The hot path is still type-safe; only the dispatch boundary is dynamic.

---

## Key features

- **Rank-specialized kernels**: matmul, transpose, and elementwise ops with dedicated kernels for 2D tensors (tiled shared memory, vectorized loads)
- **N-dimensional support**: generalized kernels for arbitrary-rank tensors with stride-aware index computation
- **RAII GPU memory**: `DeviceTensor<T>` manages allocation/deallocation with move semantics and no raw pointer leaks
- **Automatic kernel dispatch**: runtime selection of the most efficient kernel based on tensor rank and operation type
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