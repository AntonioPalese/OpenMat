# Fused Operations in OpenMat

## What is a fused operation

A **fused operation** is a single kernel that performs several consecutive transformations on a tensor in one pass over memory, avoiding the materialization of intermediate results. `(x + a) * b` in its non-fused form requires two kernels and two round-trips to global memory; fused, the same kernel reads once and writes once.

The advantage on GPU is that element-wise operations are almost always **memory-bound**: the GPU computes arithmetic far faster than it can move data from DRAM. Fusing reduces pressure on global memory bandwidth, and — just as importantly for this library — it removes an intermediate `Tensor` allocation from the stream-ordered pool.

---

## Architecture

Three pieces, all in [`headers/ops/kernels/fused_op.cuh`](../headers/ops/kernels/fused_op.cuh) with kernels and instantiations in [`src/ops/kernels/fused_op.cu`](../src/ops/kernels/fused_op.cu):

1. **Functors** — small trivially-copyable structs with an `operator()`.
2. **Combinators** — `Compose` and `BinaryCompose`, which chain functors into one.
3. **Launchers** — `launch_apply_op` and `launch_apply_binary_op`, rank-specialized kernels that apply any functor element-wise.

`Tensor<T>` then exposes the whole thing through `apply` / `apply_binary` and a set of named shorthands.

### Every functor is `__host__ __device__`

This is the single most important property of the design. Each functor is annotated `__host__ __device__`, so **the same functor object drives both the CUDA kernel and the CPU loop**:

```cpp
template <typename T>
struct Add {
    T a;
    __host__ __device__ T operator()(T x) const { return x + a; }
};
```

`Tensor::apply` branches on the device and reuses the identical `op` in either path ([`headers/tensor.inl`](../headers/tensor.inl)):

```cpp
Tensor<value_type> out(this->shape(), this->device(), Stream(s.get()));
if (this->device_type() == DEVICE_TYPE::CPU) {
    auto src = this->view();
    auto dst = out.view();
    size_t n = src.size();
    for (size_t i = 0; i < n; ++i)
        dst[i] = op(src[i]);
} else {
    launch_apply_op<value_type>(this->view(), out.view(), op, s.get());
}
```

There is no second CPU implementation to keep in sync — which is why the `CPUGPUConsistency_*` tests in [`tests/test_fused_ops.cpp`](../tests/test_fused_ops.cpp) are meaningful rather than tautological.

---

## The functors

### Unary, scalar-capturing

| Functor | Body | Instantiated |
|---|---|---|
| `Add<T>{a}` | `x + a` | ✅ |
| `Mul<T>{b}` | `x * b` | ✅ |
| `Div<T>{b}` | `x / b` | ✅ |
| `Pow<T>{b}` | `pow(x, b)` | ❌ — see below |

### Unary, stateless

| Functor | Body | Instantiated |
|---|---|---|
| `ReLU<T>` | `float(x) > 0 ? x : T(0)` | ✅ |
| `Sigmoid<T>` | `1 / (1 + expf(-float(x)))` | ✅ |

Both route through `float` deliberately: it is what makes them work unchanged for `float16_t`, whose comparison and `exp` are not defined directly.

### Binary

`BinaryAdd<T>`, `BinarySub<T>`, `BinaryMul<T>`, `BinaryDiv<T>` — stateless, `operator()(T x, T y)`. All instantiated.

### Composition

```cpp
template <typename F, typename G>
struct Compose {
    F f;
    G g;
    template <typename T>
    __host__ __device__ auto operator()(T x) const -> decltype(g(f(x))) { return g(f(x)); }
};
```

`Compose{f, g}` is **`g(f(x))`** — `f` applies first. So `Compose<Mul<T>, Add<T>>{Mul{s}, Add{k}}` is `x * s + k`.

Note the signature: a template `operator()` with a **trailing `decltype` return type**. This is C++17. An earlier version used `auto operator()(auto x)` — an abbreviated function template, which is C++20 — and the project sets `CMAKE_CXX_STANDARD 17` / `CMAKE_CUDA_STANDARD 17` ([`CMakeLists.txt`](../CMakeLists.txt):42–43). Keep the trailing-return form when adding combinators.

`BinaryCompose<BinOp, UnaryOp>` chains a binary op with a unary post-op: `post(bin(x, y))`.

---

## The launchers

```cpp
template<typename T, typename Op>
void launch_apply_op(const TensorView<const T> src, TensorView<T> dst,
                     Op op, cudaStream_t stream = 0);

template<typename T, typename Op>
void launch_apply_binary_op(const TensorView<const T> lhs, const TensorView<const T> rhs,
                            TensorView<T> dst, Op op, cudaStream_t stream = 0);
```

Both switch on `dst.rank` and pick a rank-tuned launch configuration, matching the layouts used by the binary-op macros elsewhere in the codebase:

| Rank | Block | Grid |
|---|---|---|
| 1 | `dim3(256)` | `ceil(shape[0] / 256)` |
| 2 | `dim3(16, 16)` | `ceil(shape[1]/16), ceil(shape[0]/16)` |
| 3 | `dim3(8, 8, 8)` | `ceil(shape[2]/8), ceil(shape[1]/8), ceil(shape[0]/8)` |
| 4 | `dim3(8, 8, min(shape[1], 8))` | `ceil(shape[3]/tx), ceil(shape[2]/ty), shape[0]` |
| ≥ 5 | `dim3(256)` | `ceil(size() / 256)` — flat `_nd` kernel |

The rank-4 kernel grid-strides over the channel dimension (`for c = threadIdx.z; c < shape[1]; c += blockDim.z`) rather than mapping it one-thread-per-element, so it handles channel counts above 8. The `_nd` fallback reconstructs the multi-index from a linear index.

**Synchronization.** Both launchers end with:

```cpp
CUDA_CHECK;
if (stream == nullptr) cudaDeviceSynchronize();
```

So the default-stream form is synchronous and safe to read from immediately; passing a real stream makes the call asynchronous and hands synchronization to the caller. This is the same convention as every other launcher in the library.

**`src` must be `TensorView<const T>`, not `TensorView<T>`.** `match()` takes a `TensorView<T>`, so the shape check has to be written `src.match(dst)`. Writing `dst.match(src)` attempts an illegal `const T*` → `T*` conversion.

---

## `Tensor<T>` surface

```cpp
// Generic
template<typename Op> Tensor<T> apply(Op op) const;
template<typename Op> Tensor<T> apply(Op op, const Stream& s) const;
template<typename Op> Tensor<T> apply_binary(const Tensor<T>& rhs, Op op) const;

// Named unary
Tensor<T> relu() const;                      Tensor<T> relu(const Stream& s) const;
Tensor<T> sigmoid() const;                   Tensor<T> sigmoid(const Stream& s) const;
Tensor<T> scale_shift(T scale, T shift) const;   // x * scale + shift
Tensor<T> shift_scale(T shift, T scale) const;   // (x + shift) * scale

// Named binary
Tensor<T> fused_add_mul(const Tensor<T>& rhs, T scale) const;  // (a + b) * scale
Tensor<T> fused_sub_mul(const Tensor<T>& rhs, T scale) const;  // (a - b) * scale
Tensor<T> fused_mul_add(const Tensor<T>& rhs, T shift) const;  // (a * b) + shift
Tensor<T> fused_div_add(const Tensor<T>& rhs, T shift) const;  // (a / b) + shift
```

Each named method is a one-liner over the generic one — for example:

```cpp
Tensor<T> Tensor<T>::scale_shift(T scale, T shift) const {
    return this->apply(Compose<Mul<T>, Add<T>>{Mul<T>{scale}, Add<T>{shift}});
}

Tensor<T> Tensor<T>::fused_add_mul(const Tensor<T>& rhs, T scale) const {
    return this->apply_binary(rhs, BinaryCompose<BinaryAdd<T>, Mul<T>>{BinaryAdd<T>{}, Mul<T>{scale}});
}
```

`apply_binary` validates shapes on the CPU path and throws `std::invalid_argument` on mismatch.

### Stream coverage is asymmetric

`apply` has a `(Op, const Stream&)` overload; `relu` and `sigmoid` have stream siblings that forward to it. **`apply_binary` does not** — it allocates its output with the default stream and calls `launch_apply_binary_op` without a stream argument, so `scale_shift`, `shift_scale`, and all four `fused_*_*` methods are synchronous default-stream calls. Adding a stream overload for `apply_binary` (and the four `fused_*` shorthands) is the natural next step, and would follow the same pattern as `apply`: take `const Stream& s`, build the output with the private `Tensor(shape, device, Stream)` constructor, and pass `s.get()` down to the launcher.

---

## Explicit instantiations — the one thing that will bite you

`fused_op.cu` is a CUDA translation unit; `tensor.inl` is consumed by `.cpp` translation units, which cannot instantiate CUDA templates. **Every `(T, Op)` pair reachable from a `.cpp` must be explicitly instantiated in `fused_op.cu`, or you get a link error.** Calls made from a `.cu` file instantiate implicitly and hide the problem entirely — so a functor can look fine in a CUDA test and fail to link from the library.

Currently instantiated for `float`, `int`, `char`, `float16_t`:

**Unary** (individually listed) — `Add`, `Mul`, `Div`, `Compose<Mul, Add>`, `Compose<Add, Mul>`, `ReLU`, `Sigmoid`.

**Binary** (via the `INSTANTIATE_BINARY_FUSED(T)` macro) — `BinaryAdd`, `BinarySub`, `BinaryMul`, `BinaryDiv`, `BinaryCompose<BinaryAdd, Mul>`, `BinaryCompose<BinarySub, Mul>`, `BinaryCompose<BinaryMul, Add>`, `BinaryCompose<BinaryDiv, Add>`.

**Not instantiated:** `Pow<T>` is defined in the header but has no explicit instantiation and no `Tensor` method. Using it from a `.cpp` is a link error today. `double` is absent throughout — it has no GPU instantiation anywhere in the library.

Note that the instantiation list is a list of *exact* types: `Compose<Mul<float>, Add<float>>` being instantiated says nothing about `Compose<Add<float>, Div<float>>`.

---

## How to extend

### Add a new functor

```cpp
template <typename T>
struct Tanh {
    __host__ __device__ T operator()(T x) const {
        return static_cast<T>(tanhf(static_cast<float>(x)));
    }
};
```

Two rules:

- **Annotate `__host__ __device__`**, or the CPU path in `apply` will not compile.
- **Use `static_cast<T>(0)`, never `T{0}`**, for types with non-trivial constructors such as `float16_t`. nvcc forbids dynamic initialization of `__shared__` variables, and `T{0}` invokes the constructor rather than performing a plain zero-cast.

Then add the instantiations in `fused_op.cu`:

```cpp
template void launch_apply_op<float>    (const TensorView<const float>,     TensorView<float>,     Tanh<float>,     cudaStream_t);
template void launch_apply_op<int>      (const TensorView<const int>,       TensorView<int>,       Tanh<int>,       cudaStream_t);
template void launch_apply_op<char>     (const TensorView<const char>,      TensorView<char>,      Tanh<char>,      cudaStream_t);
template void launch_apply_op<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, Tanh<float16_t>, cudaStream_t);
```

...plus any `Compose` combination you intend to reach from a `.cpp`.

### Add a `Tensor` method

Follow `relu`: implement the `(…, const Stream&)` overload first and make the no-stream form a one-line delegate to `Stream::default_stream()`. This is the single-source-of-truth invariant the whole of `tensor.inl` is built on.

### Expose it to Python

Three edits, per [`CLAUDE.md`](../CLAUDE.md): the function in [`src/python/openmat_capi_impl.inc`](../src/python/openmat_capi_impl.inc) (written once — both dtypes get it), the `ctypes` `restype`/`argtypes` in `_declare_dtype()` in [`python/openmat/_clib.py`](../python/openmat/_clib.py), and the wrapper in [`python/openmat/tensor.py`](../python/openmat/tensor.py).

---

## Tests

[`tests/test_fused_ops.cpp`](../tests/test_fused_ops.cpp) — 36 tests, run as `./build/tests/test_fused_ops`:

- CPU correctness on known values for `apply`, `apply_binary`, `scale_shift`, `relu`, `sigmoid`
- GPU correctness at rank 1, 2 and 3 for each functor and each named method
- CPU↔GPU consistency (`CPUGPUConsistency_ScaleShift`, `_FusedAddMul`, `CPUGPUReLU_Consistency`, `CPUGPUSigmoid_Consistency`)
- Equivalence against the unfused two-step computation (`FusedAddMulEquivalent`, `FusedMulAddEquivalent`, `ScaleShiftEquivalent`)
- `ShapeMismatchThrows`

The equivalence tests are the ones that matter when adding a functor: they pin the fused result to the same value the naive path produces.

---

## Status summary

| Aspect | Status |
|---|---|
| `launch_apply_op` — rank 1–4 + nd | ✅ Implemented |
| `launch_apply_binary_op` — rank 1–4 + nd | ✅ Implemented |
| Unary functors `Add`, `Mul`, `Div`, `ReLU`, `Sigmoid` | ✅ Defined and instantiated |
| Binary functors `BinaryAdd/Sub/Mul/Div` | ✅ Defined and instantiated |
| `Compose` / `BinaryCompose` | ✅ Defined, C++17-clean, instantiated for the pairs in use |
| CPU execution path | ✅ Same functors, no duplicate implementation |
| Integration in `Tensor<T>` | ✅ `apply`, `apply_binary` + 8 named methods |
| Stream overloads | ⚠️ `apply` / `relu` / `sigmoid` only — `apply_binary` and the `fused_*` family are default-stream |
| `Pow<T>` | ⚠️ Defined, not instantiated, not exposed on `Tensor` |
| `double` support | ❌ No GPU instantiation anywhere in the library |
| Python bindings | ✅ `relu`, `sigmoid`, `scale_shift`, `shift_scale`, `fused_*` |
| Test coverage | ✅ 36 tests in `test_fused_ops` |
