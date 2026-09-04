#pragma once
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#include "cuda_defines.cuh"
#include "type_traits/types.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Contiguous fast path for elementwise kernels
//
// The rank-specialized launchers map tensor axes onto grid axes, which means a
// warp's 32 lanes only stay contiguous in memory for rank 1. At rank 2 a
// dim3(16,16) block gives each warp two disjoint runs of 16 elements; at rank 3
// a dim3(8,8,8) block gives it four runs of 8. Those are 64- and 32-byte
// requests against a 128-byte cache line, and they cost real bandwidth — on the
// reference GB10, `add` over 16 M floats runs at 222 GB/s at rank 1, 185 GB/s
// at rank 2 and 133 GB/s at rank 3, for exactly the same traffic.
//
// Every tensor OpenMat produces today is contiguous row-major (reshape and
// friends deep-copy, nothing returns an aliasing view), so the axis structure
// carries no information the kernel needs: the whole tensor is one flat run.
// This path throws the shape away, indexes the buffer linearly and recovers the
// rank-1 layout for every rank. `is_contiguous()` is what gates it, so the day
// views arrive (roadmap P2) a strided view simply falls back to the existing
// rank-specialized kernels instead of silently reading the wrong elements.
//
// Per-thread width is 4 bytes, not 16. Vector loads are the usual advice, and
// they measured *worse* here: with one 16-byte float4 per thread `add` drops to
// 216 GB/s against 235 GB/s for one scalar float, because the launch loses the
// thread-level parallelism it needs to keep the memory pipeline full. What does
// matter is that no thread moves *less* than 4 bytes — a warp of `char` lanes
// requests 32 bytes and reaches only 193 GB/s. So the pack width is
// `4 / sizeof(T)`: 1 for float and int (already right), 2 for float16_t,
// 4 for char. Measured on 16 M elements, 3× traffic, GB10:
//
//     float  V=1   235 GB/s      float16_t V=1  235 GB/s   V=2  239 GB/s
//     int    V=1   239 GB/s      char      V=1  193 GB/s   V=4  236 GB/s
//     float  V=4   216 GB/s      char      V=16 223 GB/s
//
// Block size is 256 for the same reason it is elsewhere in the library. 512 and
// 1024 buy 2-3 % once the working set passes L2 (24 MiB here) and lose up to
// 35 % below it, where occupancy rather than bandwidth is the limit; 256 is
// never worse than 2.5 % off the best. Re-measure before changing it.
//
// A grid-stride loop was also tried and is not used: capping the grid to a few
// waves per SM costs 8-12 %, and at the exact block count the loop body runs
// once and buys nothing over a plain bounds check.
// ─────────────────────────────────────────────────────────────────────────────

namespace om {
namespace detail {

    // Elements per thread: enough of them that a lane always moves 4 bytes.
    template <typename T>
    struct pack_width {
        static constexpr int value = sizeof(T) >= 4 ? 1 : static_cast<int>(4 / sizeof(T));
    };

    constexpr unsigned int CONTIG_BLOCK = 256;

    inline bool contig_aligned(const void* p, size_t bytes)
    {
        return (reinterpret_cast<uintptr_t>(p) % bytes) == 0;
    }

} // namespace detail
} // namespace om

// The kernel bodies and the launch statements below are only parseable by nvcc.
// tensor.cuh pulls this header into plain .cpp translation units (the Python
// C-ABI layer among them), where __global__ expands to nothing and blockIdx
// does not exist at all.
#if defined(__CUDACC__)

namespace om {
namespace detail {

    // V > 1 packs V elements into one 4-byte word so the lane issues a single
    // load. The punning goes through memcpy rather than a union or a struct
    // assignment: float16_t has a user-provided constructor, so a union of it is
    // ill-formed and a member-wise copy is free to lower to two 2-byte accesses,
    // which is the whole thing this path exists to avoid.
    template <typename T, int V, typename Op>
    __global__ void contig_binary_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                         T* __restrict__ dst, size_t n_pack, size_t n, Op op)
    {
        const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

        if constexpr (V == 1) {
            if (i < n_pack)
                dst[i] = op(device_load(&a[i]), device_load(&b[i]));
        } else {
            using word = unsigned int;
            if (i < n_pack) {
                const word wa = __ldg(reinterpret_cast<const word*>(a) + i);
                const word wb = __ldg(reinterpret_cast<const word*>(b) + i);
                T xa[V], xb[V], r[V];
                memcpy(xa, &wa, sizeof(word));
                memcpy(xb, &wb, sizeof(word));
#pragma unroll
                for (int j = 0; j < V; ++j) r[j] = op(xa[j], xb[j]);
                word wr;
                memcpy(&wr, r, sizeof(word));
                reinterpret_cast<word*>(dst)[i] = wr;
            }
            // Fewer than V elements can be left over. One block picks them up;
            // the branch is uniformly false in every other block.
            const size_t t = n_pack * V + i;
            if (blockIdx.x == 0 && t < n)
                dst[t] = op(a[t], b[t]);
        }
    }

    template <typename T, int V, typename Op>
    __global__ void contig_unary_kernel(const T* __restrict__ src, T* __restrict__ dst,
                                        size_t n_pack, size_t n, Op op)
    {
        const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

        if constexpr (V == 1) {
            if (i < n_pack)
                dst[i] = op(device_load(&src[i]));
        } else {
            using word = unsigned int;
            if (i < n_pack) {
                const word ws = __ldg(reinterpret_cast<const word*>(src) + i);
                T xs[V], r[V];
                memcpy(xs, &ws, sizeof(word));
#pragma unroll
                for (int j = 0; j < V; ++j) r[j] = op(xs[j]);
                word wr;
                memcpy(&wr, r, sizeof(word));
                reinterpret_cast<word*>(dst)[i] = wr;
            }
            const size_t t = n_pack * V + i;
            if (blockIdx.x == 0 && t < n)
                dst[t] = op(src[t]);
        }
    }

    template <typename T, int V>
    __global__ void contig_fill_kernel(T* __restrict__ dst, T value, size_t n_pack, size_t n)
    {
        const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

        if constexpr (V == 1) {
            if (i < n_pack) dst[i] = value;
        } else {
            using word = unsigned int;
            if (i < n_pack) {
                T r[V];
#pragma unroll
                for (int j = 0; j < V; ++j) r[j] = value;
                word wr;
                memcpy(&wr, r, sizeof(word));
                reinterpret_cast<word*>(dst)[i] = wr;
            }
            const size_t t = n_pack * V + i;
            if (blockIdx.x == 0 && t < n) dst[t] = value;
        }
    }

    // ── Host launchers ──────────────────────────────────────────────────────
    //
    // Each returns the name of the kernel it launched, for CUDA_CHECK_LAUNCH,
    // or nullptr when it declined — the caller then falls through to its
    // existing rank-specialized path. Declining is not an error: it is how an
    // empty tensor, an unaligned buffer or an unrepresentable grid keeps the
    // old, always-correct behaviour.

    inline bool contig_grid(size_t n, int v, size_t& n_pack, unsigned int& blocks)
    {
        n_pack = n / static_cast<size_t>(v);
        size_t gx = (n_pack + CONTIG_BLOCK - 1) / CONTIG_BLOCK;
        if (gx == 0) gx = 1;                 // only the sub-V tail exists
        if (!grid_fits(gx, 1, 1)) return false;
        blocks = static_cast<unsigned int>(gx);
        return true;
    }

    template <typename T, typename Op>
    inline const char* launch_contiguous_binary(const T* a, const T* b, T* dst, size_t n,
                                                Op op, cudaStream_t stream)
    {
        constexpr int V = pack_width<T>::value;
        if (n == 0) return nullptr;

        size_t n_pack; unsigned int blocks;

        if constexpr (V > 1) {
            constexpr size_t word_bytes = sizeof(T) * V;
            if (contig_aligned(a, word_bytes) && contig_aligned(b, word_bytes) &&
                contig_aligned(dst, word_bytes) && contig_grid(n, V, n_pack, blocks)) {
                contig_binary_kernel<T, V, Op><<<blocks, CONTIG_BLOCK, 0, stream>>>(a, b, dst, n_pack, n, op);
                return "contig_binary_kernel<packed>";
            }
        }
        if (!contig_grid(n, 1, n_pack, blocks)) return nullptr;
        contig_binary_kernel<T, 1, Op><<<blocks, CONTIG_BLOCK, 0, stream>>>(a, b, dst, n_pack, n, op);
        return "contig_binary_kernel<scalar>";
    }

    template <typename T, typename Op>
    inline const char* launch_contiguous_unary(const T* src, T* dst, size_t n,
                                               Op op, cudaStream_t stream)
    {
        constexpr int V = pack_width<T>::value;
        if (n == 0) return nullptr;

        size_t n_pack; unsigned int blocks;

        if constexpr (V > 1) {
            constexpr size_t word_bytes = sizeof(T) * V;
            if (contig_aligned(src, word_bytes) && contig_aligned(dst, word_bytes) &&
                contig_grid(n, V, n_pack, blocks)) {
                contig_unary_kernel<T, V, Op><<<blocks, CONTIG_BLOCK, 0, stream>>>(src, dst, n_pack, n, op);
                return "contig_unary_kernel<packed>";
            }
        }
        if (!contig_grid(n, 1, n_pack, blocks)) return nullptr;
        contig_unary_kernel<T, 1, Op><<<blocks, CONTIG_BLOCK, 0, stream>>>(src, dst, n_pack, n, op);
        return "contig_unary_kernel<scalar>";
    }

    template <typename T>
    inline const char* launch_contiguous_fill(T* dst, T value, size_t n, cudaStream_t stream)
    {
        constexpr int V = pack_width<T>::value;
        if (n == 0) return nullptr;

        size_t n_pack; unsigned int blocks;

        if constexpr (V > 1) {
            constexpr size_t word_bytes = sizeof(T) * V;
            if (contig_aligned(dst, word_bytes) && contig_grid(n, V, n_pack, blocks)) {
                contig_fill_kernel<T, V><<<blocks, CONTIG_BLOCK, 0, stream>>>(dst, value, n_pack, n);
                return "contig_fill_kernel<packed>";
            }
        }
        if (!contig_grid(n, 1, n_pack, blocks)) return nullptr;
        contig_fill_kernel<T, 1><<<blocks, CONTIG_BLOCK, 0, stream>>>(dst, value, n_pack, n);
        return "contig_fill_kernel<scalar>";
    }

} // namespace detail
} // namespace om

#endif // __CUDACC__
