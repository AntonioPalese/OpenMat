---
name: debug-cuda-async
description: Localize asynchronous CUDA kernel errors in OpenMat with OPENMAT_DEBUG_SYNC and compute-sanitizer. Use when a test fails with an illegal memory access, a CUDA error surfaces in an unrelated call, or a kernel launch needs to be traced to its call site.
---

# Debugging asynchronous kernel errors

`OPENMAT_DEBUG_SYNC=1` forces a `cudaStreamSynchronize` after every kernel launch, so an out-of-bounds access is reported at the launching call site with the kernel's name instead of surfacing later as an illegal access in an unrelated call:

```bash
OPENMAT_DEBUG_SYNC=1 ./build/tests/test_streams          # no rebuild needed
```

```
[CUDA ASYNC ERROR] kernel 'add_kernel_rank1' at src/ops/kernels/binary_ops.cu:10
  in void om::launch_add(...) [with T = float; ...]
  on stream 0xb0149ef4a290
  → an illegal memory access was encountered
```

`cmake -DOM_DEBUG_SYNC=ON` makes it the build's default (the env var still overrides either way, so `OPENMAT_DEBUG_SYNC=0` turns it back off); `cmake -DOM_NO_DEBUG_SYNC=ON` compiles it out entirely. It serializes streams — diagnostic mode, never a default. It complements `compute-sanitizer`: cheap enough to leave on for a whole test run, but it only localizes, it does not tell you which access was bad.

`compute-sanitizer --tool memcheck --leak-check full` is the next step, and the only thing that reports a stream-ownership violation near the call site instead of as an illegal access in an unrelated kernel later on. The gpu CI job runs it over `test_stress`, `test_allocator_stream`, `test_streams`, `test_views` and `test_dlpack`. Under Python it reports a CUDA tensor that a kernel has read as leaked even when it was freed (open, see CLAUDE.md's CI section): compare against a baseline before trusting a Python-side leak.

The `CUDA_CHECK_LAUNCH` rule that makes this machinery work is in the root [CLAUDE.md](../../../CLAUDE.md) — it applies to every kernel you write, not just to debugging.
