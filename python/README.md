# openmat

Python bindings for [OpenMat](https://github.com/AntonioPalese/OpenMat) — a CUDA tensor library.

## Requirements

- NVIDIA GPU (compute capability ≥ 7.0)
- CUDA Toolkit ≥ 12.0
- The compiled `OpenMat.so` (build with `./compile.sh` in the repo root)

## Install (editable, development)

```bash
# from repo root
cd python
uv sync
uv pip install -e .
```

## Quick start

```python
from openmat import Tensor

a = Tensor.from_list([1, 2, 3, 4], [2, 2])
b = Tensor.ones([2, 2])

c = (a + b).sum()          # 14.0
g = a.cuda()               # move to GPU
g2 = (g * 2).reshape([4]) # element-wise, then flatten
print(g2.numpy())          # back to numpy
```
