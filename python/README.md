# openmat

Python bindings for [OpenMat](https://github.com/AntonioPalese/OpenMat) — a CUDA tensor library.

A **ctypes** binding (not pybind) over the C-ABI layer compiled into `OpenMat.so`.
The surface mirrors `om::Tensor<T>`, including the CUDA streams that every
operation actually runs on.

## Requirements

- NVIDIA GPU, CUDA Toolkit ≥ 11.2
- The compiled `OpenMat.so` (build with `./compile.sh` in the repo root)
- Python ≥ 3.9; numpy is optional (only `numpy()` / `from_numpy()` need it)

## Install

```bash
cd python
uv sync
uv pip install -e .            # bundles build/OpenMat.so into the package
```

Or skip installing entirely — `openmat/_clib.py` finds the library at
`$OPENMAT_LIB`, then `openmat/OpenMat.so`, then `<repo>/build/OpenMat.so`, so a
source checkout works straight after `./compile.sh`:

```bash
OPENMAT_LIB=build/OpenMat.so PYTHONPATH=python python -c "import openmat"
cd python && pytest
```

## Quick start

```python
import openmat as om
from openmat import Tensor

a = Tensor([[1, 2], [3, 4]])          # nested lists, flat lists or ndarrays
b = om.ones([2, 2])

(a + b).sum()                          # 14.0
(a @ b).tolist()                       # [[3.0, 3.0], [7.0, 7.0]]
a.T.tolist()                           # [[1.0, 3.0], [2.0, 4.0]]
a[1, 0]; a[0, 0] = 9                   # element access, bounds-checked
g = a.cuda()                           # move to the GPU
g.scale_shift(2.0, 1.0).numpy()        # x*2+1 in one kernel, then back to numpy
```

## dtypes

`float32` (the default) and `int32`.  Each is a separate family of exported
symbols (`om_tensor_float_*`, `om_tensor_int_*`) generated from one C++ body.

```python
i = Tensor([1, 2, 3], dtype="int32")
i.dtype                                # dtype('int32')
i.astype("float32")                    # host round-trip conversion
Tensor.ones([2]) + i                   # TypeError: dtype mismatch
```

## Streams

Streams are the canonical execution path: the synchronous API is a delegate over
the stream API with the default (null) stream.  Stream overloads **enqueue** work
and return immediately; you synchronize before reading.

```python
from openmat import Stream

with Stream() as s:                    # synchronizes on exit
    c = a.add(b, stream=s).mul(2.0, stream=s)
print(c.tolist())

s = Stream()
d = a.cuda(stream=s)
s.synchronize()
```

`stream=` is accepted by `add`, `sub`, `mul`, `div`, `matmul`, `transpose`,
`permute`, `relu`, `sigmoid`, `cpu`, `cuda`, `to`, `fill`, `Tensor.from_list`,
and every member of the in-place / `_out` families below.

Memory a tensor got from a stream's pool must be freed on that same stream, so
every result holds a reference on the stream it was produced on.  The reference
count lives in the C layer, which is why `Stream.close()` is safe even while
tensors from that stream are still alive — the `cudaStream_t` goes away when the
last of them does.

## In-place ops and `out=`

Every op above allocates its result. In a loop that is one allocation and one
free per op per iteration, and a peak footprint equal to the whole expression.
The trailing-underscore family writes back into the tensor it is called on, and
the `_out` family into a destination you already own; both return the tensor
they wrote to, so calls chain.

```python
w = om.Tensor.zeros([1024], device="cuda")
g = om.Tensor.full([1024], 0.5, device="cuda")

for _ in range(steps):
    w.add_(g)                   # w's buffer never moves
    w.relu_()

w -= g                          # += -= *= /= are the same thing
w *= 0.9

out = om.Tensor.zeros([1024], device="cuda")
for batch in batches:           # one destination for the whole loop
    a.mul_out(b, out)
```

In-place: `add_`, `sub_`, `mul_`, `div_` (tensor or scalar), `relu_`,
`sigmoid_`, `fill_`, and `+= -= *= /=`.

Destination-provided: `add_out`, `sub_out`, `mul_out`, `div_out`, `relu_out`,
`sigmoid_out`, `matmul_out`, `transpose_out`, `permute_out`. `out` must already
have the result's shape, dtype and device.

`matmul`, `transpose` and `permute` have no in-place form and reject a
destination that is one of their operands — their kernels read elements they do
not write, so the answer would be wrong. The elementwise family is free to
alias, which is exactly what `add_` is.

One caveat when you pass both `out` and `stream`: memory a tensor got from a
stream's pool must be used in an order that stream can see, so enqueueing into a
destination allocated on a *different* stream is only correct once you have
ordered the two yourself. The allocating forms cannot hit this — they allocate
on the stream they run on.

## API

| group | members |
|---|---|
| factories | `Tensor(data)`, `zeros`, `ones`, `full`, `empty`, `arange`, `from_list`, `from_numpy` |
| metadata | `shape`, `stride`, `rank`/`ndim`, `size`, `dtype`, `itemsize`, `nbytes`, `device`, `device_index`, `is_cuda`, `stream`, `data_ptr()` |
| data | `numpy()`, `tolist()`, `flat()`, `item()`, `fill()`, `copy()`, `t[i, j]`, `t[i, j] = v` |
| device | `cpu()`, `cuda()`, `to(device)`, `astype()`, `synchronize()` |
| arithmetic | `+ - * / @`, reflected and scalar forms, `add`/`sub`/`mul`/`div`/`matmul` |
| in-place | `add_`, `sub_`, `mul_`, `div_`, `relu_`, `sigmoid_`, `fill_`, `+= -= *= /=` |
| destination-provided | `add_out`, `sub_out`, `mul_out`, `div_out`, `relu_out`, `sigmoid_out`, `matmul_out`, `transpose_out`, `permute_out` |
| reductions | `sum()`, `mean()`, `min()`, `max()` |
| shape | `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`/`T`, `permute` |
| fused | `relu`, `sigmoid`, `scale_shift`, `shift_scale`, `fused_add_mul`, `fused_sub_mul`, `fused_mul_add`, `fused_div_add` |
| module | `om.cuda_is_available()`, `om.device_count()`, `om.synchronize()`, `om.dtype()` |

Host tensors expose `__array_interface__`, so `np.asarray(t)` is a zero-copy view
that keeps the tensor alive; `t.numpy()` always copies.  CUDA tensors expose
`__cuda_array_interface__` for interop with cupy or torch.
