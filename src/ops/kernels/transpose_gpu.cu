#include "ops/kernels/transpose_gpu.cuh"
#include "type_traits/types.cuh"

namespace om
{

// ── 2D transpose kernel ──────────────────────────────────────────────────────
// Uses shared-memory tiling to coalesce both reads and writes.
// Tile size 32×32 with a +1 padding column to avoid bank conflicts.
namespace {

constexpr int TILE = 32;

template<typename T>
__global__ void transpose_kernel(
    const DeviceTensorView<const T> src,
    DeviceTensorView<T>             dst)
{
    __shared__ T tile[TILE][TILE + 1];

    size_t src_col = blockIdx.x * TILE + threadIdx.x;
    size_t src_row = blockIdx.y * TILE + threadIdx.y;
    size_t M = src.shape[0]; // rows of src
    size_t N = src.shape[1]; // cols of src

    if (src_row < M && src_col < N)
        tile[threadIdx.y][threadIdx.x] = src(src_row, src_col);

    __syncthreads();

    // Write transposed block: dst is N×M
    size_t dst_col = blockIdx.y * TILE + threadIdx.x;
    size_t dst_row = blockIdx.x * TILE + threadIdx.y;

    if (dst_row < N && dst_col < M)
        dst(dst_row, dst_col) = tile[threadIdx.x][threadIdx.y];
}

// Trivially-copyable wrapper so the axes array is passed by value to CUDA kernels
// (raw C arrays decay to pointers when used as kernel parameters).
struct AxesBuf {
    size_t v[MAX_RANK] = {};
};

// ── N-D permute kernel ───────────────────────────────────────────────────────
// Flat thread per output element; reconstructs multi-index, applies perm.
// axes is passed by value (AxesBuf struct) — no device allocation needed.
template<typename T>
__global__ void permute_kernel(
    const DeviceTensorView<const T> src,
    DeviceTensorView<T>             dst,
    AxesBuf                         axes,
    size_t                          rank)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = dst.size();
    if (idx >= total) return;

    // Decompose flat dst index into dst multi-index
    size_t dst_idx[MAX_RANK];
    size_t tmp = idx;
    for (int d = (int)rank - 1; d >= 0; --d) {
        dst_idx[d] = tmp % dst.shape[d];
        tmp        /= dst.shape[d];
    }

    // Map to src multi-index via inverse permutation:
    // dst axis d came from src axis axes[d], so src_idx[axes[d]] = dst_idx[d]
    size_t src_idx[MAX_RANK];
    for (size_t d = 0; d < rank; ++d)
        src_idx[axes.v[d]] = dst_idx[d];

    // Compute flat src offset
    size_t src_flat = 0;
    for (size_t d = 0; d < rank; ++d)
        src_flat += src_idx[d] * src.stride[d];

    dst[idx] = src.data[src_flat];
}

} // anonymous namespace

// ── launch_transpose ─────────────────────────────────────────────────────────
template<typename T>
void launch_transpose(const TensorView<const T> src, TensorView<T> dst, cudaStream_t stream)
{
    if (src.rank != 2 || dst.rank != 2)
        throw std::runtime_error("launch_transpose: tensors must be rank-2");
    if (src.shape[0] != dst.shape[1] || src.shape[1] != dst.shape[0])
        throw std::runtime_error("launch_transpose: dst shape must be transposed src shape");

    size_t M = src.shape[0];
    size_t N = src.shape[1];

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    transpose_kernel<T><<<blocks, threads, 0, stream>>>(src.as_device_tw(), dst.as_device_tw());
    CUDA_CHECK;
    if (stream == nullptr) cudaDeviceSynchronize();
}

// ── launch_permute ───────────────────────────────────────────────────────────
template<typename T>
void launch_permute(const TensorView<const T> src, TensorView<T> dst,
                    const size_t* h_axes, size_t rank, cudaStream_t stream)
{
    // Copy axes into a trivially-copyable struct so the kernel receives them by
    // value — no device allocation required.
    AxesBuf axes_buf;
    for (size_t i = 0; i < rank; ++i) axes_buf.v[i] = h_axes[i];

    size_t total = dst.size();
    dim3 threads(256);
    dim3 blocks((total + 255) / 256);

    permute_kernel<T><<<blocks, threads, 0, stream>>>(src.as_device_tw(), dst.as_device_tw(), axes_buf, rank);
    CUDA_CHECK;
    if (stream == nullptr) cudaDeviceSynchronize();
}

// ── explicit instantiations ──────────────────────────────────────────────────
template void launch_transpose<float>    (const TensorView<const float>,     TensorView<float>,     cudaStream_t);
template void launch_transpose<double>   (const TensorView<const double>,    TensorView<double>,    cudaStream_t);
template void launch_transpose<int>      (const TensorView<const int>,       TensorView<int>,       cudaStream_t);
template void launch_transpose<char>     (const TensorView<const char>,      TensorView<char>,      cudaStream_t);
template void launch_transpose<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, cudaStream_t);

template void launch_permute<float>    (const TensorView<const float>,     TensorView<float>,     const size_t*, size_t, cudaStream_t);
template void launch_permute<double>   (const TensorView<const double>,    TensorView<double>,    const size_t*, size_t, cudaStream_t);
template void launch_permute<int>      (const TensorView<const int>,       TensorView<int>,       const size_t*, size_t, cudaStream_t);
template void launch_permute<char>     (const TensorView<const char>,      TensorView<char>,      const size_t*, size_t, cudaStream_t);
template void launch_permute<float16_t>(const TensorView<const float16_t>, TensorView<float16_t>, const size_t*, size_t, cudaStream_t);

} // namespace om
