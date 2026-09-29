---
name: add-binary-op
description: The file-by-file checklist for adding a new binary elementwise op to OpenMat (CUDA kernel, launch macros, functor, CPU side, Tensor method). Use when adding or removing an elementwise op such as add/sub/mul/div.
---

# Adding a new binary op

1. `src/ops/kernels/binary_ops.cu` — kernel bodies via `DEFINE_BINARY_OP_KERNEL_K1/K2/K3/K4/ND` + `DEFINE_BINARY_OP_LAUNCH` + `DEFINE_BINARY_OP_LAUNCH_FRW_DEC`.
2. `headers/ops/kernels/binary_op_macros.cuh` — `DEFINE_BINARY_OP_LAUNCH_H` / `DEFINE_BINARY_OP_KERNEL_H`, plus `DEFINE_BINARY_OP_FUNCTOR_H(OP, expr in a and b)` — the launch macro references `OP_fn<T>` for the contiguous fast path, so omitting it is a compile error, not a silent slow path.
3. `src/ops/cpu/binary_ops.cpp` + `headers/ops/cpu/binary_op_macros.h` — CPU side.
4. `headers/tensor.cuh` / `.inl` — the `(rhs, const Stream&)` method plus the one-line no-stream delegate.
5. Only if a stream-less free function is wanted: register in `kernel_launcher.h`/`.inl`.

Sources are picked up with `file(GLOB ...)`, so a newly added `.cpp`/`.cu` needs a CMake re-run, not just `make`.

The invariants these steps must respect — the stream overload is the real implementation, the contiguous fast path and its `grid_fits` guard, the `_out`/`_` three-form rule, and `CUDA_CHECK_LAUNCH` after every launch — are in the root [CLAUDE.md](../../../CLAUDE.md).
