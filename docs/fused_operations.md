# Fused Operations in OpenMat

## What is a fused operation

A **fused operation** is a single CUDA kernel that performs multiple consecutive transformations on a tensor in a single pass over memory, avoiding the materialization of intermediate results. For example, `(x + a) * b` in its non-fused version requires two separate kernels and two round-trips to global memory; in the fused version, the same kernel reads once and writes once.

The main advantage on GPU is that element-wise operations are almost always **memory-bound**: the GPU computes arithmetic faster than it can transfer data from DRAM. Fusing operations reduces pressure on global memory bandwidth.

---

## What is implemented today

The file [`headers/ops/kernels/fused_op.cuh`](../headers/ops/kernels/fused_op.cuh) and its implementation [`src/ops/kernels/fused_op.cu`](../src/ops/kernels/fused_op.cu) introduce the infrastructure for fused ops. The mechanism is based on three components.

### 1. Functor types — `Add<T>` and `Mul<T>`

```cpp
template <typename T>
struct Add {
    T a;
    __device__ T operator()(T x) const { return x + a; }
};

template <typename T>
struct Mul {
    T b;
    __device__ T operator()(T x) const { return x * b; }
};
```

These are `__device__` functors; each captures a scalar value and applies its own operation to a single element.

### 2. Composition with `Compose<F, G>`

```cpp
template <typename F, typename G>
struct Compose {
    F f;
    G g;
    __device__ auto operator()(auto x) const { return g(f(x)); }
};
```

`Compose` chains two functors into one: `Compose{Add{a}, Mul{b}}` is equivalent to `x → (x + a) * b`. Composition is recursive: a `Compose` can be wrapped inside another `Compose` for longer chains, without allocating any intermediate memory.

### 3. Generic kernel `launch_apply_op`

```cpp
template<typename T, typename Op>
void launch_apply_op(const TensorView<const T> src, TensorView<T> dst, Op op);
```

The kernel takes any `Op` (a functor, a composition, etc.) and applies it element-wise. As with the existing binary operations, dispatch is done by rank (1–4 with dedicated kernels, ≥5 with a flat `apply_op_nd` fallback), using the same grid layouts as the rest of the codebase.

---

## Current status and limitations

### Incomplete explicit instantiations

The `.cu` file only instantiates `launch_apply_op` for `Op = Add<T>`:

```cpp
template void launch_apply_op<float>(..., Add<float> op);
template void launch_apply_op<int>(..., Add<int> op);
// ...
```

`Mul<T>` and `Compose<F,G>` **have no explicit instantiations**. Attempting to use them from a `.cpp` translation unit (which cannot instantiate CUDA templates) will result in a linker error. To use them today, `launch_apply_op` must be called from a `.cu` file.

### No integration in `Tensor<T>`

There is no `Tensor::fused_apply(Op)` method or public API that exposes the composition mechanism. The user must manually construct the functors and call `launch_apply_op` on the view.

### Unary operations only (tensor → scalar)

Fusion covers the form `dst[i] = op(src[i])`. Fusion of **binary** operations (two tensor inputs) is not yet supported — this would require functors with two arguments and separate kernels.

### `Compose` uses `auto` as a parameter (C++20)

The signature `__device__ auto operator()(auto x)` requires C++20. The project is configured with `CMAKE_CUDA_STANDARD 17` ([`CMakeLists.txt`](../CMakeLists.txt):35–36), so `Compose` only compiles if nvcc accepts it as an extension. This is a portability risk.

---

## How to extend

### Adding a new functor

```cpp
template <typename T>
struct ReLU {
    __device__ T operator()(T x) const { return x > T{0} ? x : T{0}; }
};
```

Then add the explicit instantiations in `fused_op.cu`:

```cpp
template void launch_apply_op<float>(..., ReLU<float> op);
template void launch_apply_op<float>(..., Compose<Add<float>, ReLU<float>> op);
```

### Adding binary fusion

For a binary fusion (e.g. `dst[i] = src_a[i] + src_b[i] * scalar`) a new kernel is needed with the signature `(src_a, src_b, dst, Op)` and two-argument functors:

```cpp
template <typename F, typename G>
struct BinaryCompose {
    F f;
    G g;
    __device__ auto operator()(auto x, auto y) const { return g(f(x, y)); }
};
```

---

## Summary

| Aspect | Status |
|---|---|
| `launch_apply_op` infrastructure | Implemented, rank 1–4 + nd |
| `Add<T>`, `Mul<T>` functors | Defined |
| `Compose<F,G>` composition | Defined, not explicitly instantiated |
| Integration in `Tensor<T>` | Absent |
| Binary fusion (2 tensor inputs) | Not implemented |
| Instantiations for `Mul` and `Compose` | Missing — linker error if used from `.cpp` |