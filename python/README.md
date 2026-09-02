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
`permute`, `relu`, `sigmoid`, `cpu`, `cuda`, `to` and `Tensor.from_list`.

Memory a tensor got from a stream's pool must be freed on that same stream, so
every result holds a reference on the stream it was produced on.  The reference
count lives in the C layer, which is why `Stream.close()` is safe even while
tensors from that stream are still alive — the `cudaStream_t` goes away when the
last of them does.

## API

| group | members |
|---|---|
| factories | `Tensor(data)`, `zeros`, `ones`, `full`, `empty`, `arange`, `from_list`, `from_numpy` |
| metadata | `shape`, `stride`, `rank`/`ndim`, `size`, `dtype`, `itemsize`, `nbytes`, `device`, `device_index`, `is_cuda`, `stream`, `data_ptr()` |
| data | `numpy()`, `tolist()`, `flat()`, `item()`, `fill()`, `copy()`, `t[i, j]`, `t[i, j] = v` |
| device | `cpu()`, `cuda()`, `to(device)`, `astype()`, `synchronize()` |
| arithmetic | `+ - * / @`, reflected and scalar forms, `add`/`sub`/`mul`/`div`/`matmul` |
| reductions | `sum()`, `mean()`, `min()`, `max()` |
| shape | `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`/`T`, `permute` |
| fused | `relu`, `sigmoid`, `scale_shift`, `shift_scale`, `fused_add_mul`, `fused_sub_mul`, `fused_mul_add`, `fused_div_add` |
| module | `om.cuda_is_available()`, `om.device_count()`, `om.synchronize()`, `om.dtype()` |

Host tensors expose `__array_interface__`, so `np.asarray(t)` is a zero-copy view
that keeps the tensor alive; `t.numpy()` always copies.  CUDA tensors expose
`__cuda_array_interface__` for interop with cupy or torch.
