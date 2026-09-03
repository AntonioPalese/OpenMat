#pragma once
#include "type_traits/types.cuh"
#include <type_traits>

namespace om
{
    // Single division policy, shared by the CPU loops, the CUDA kernels and the
    // fused functors, so that both backends produce the same value.
    //
    // Floating point (float, double, float16_t): no guard at all. IEEE 754
    // already yields the right result for x/0 — +inf, -inf, and NaN for 0/0 —
    // with the correct sign. The guard this replaced (`rhs != 0 ? lhs/rhs
    // : INFINITY`) flattened -1/0 to +inf and 0/0 to +inf, matching neither
    // PyTorch nor NumPy, and its per-element cast to double ran on the fp64
    // unit (1:64 throughput on consumer GPUs).
    //
    // Integral (int, char): division by zero is undefined behaviour in both
    // C++ and CUDA, so the policy has to be explicit. x/0 returns 0, which is
    // what NumPy's integer division does; the old code returned 0 on the CPU
    // (std::numeric_limits<int>::infinity() is 0) and converted INFINITY to int
    // on the GPU, i.e. UB and two different answers.
    //
    // Deliberately still divergent: signed overflow (INT_MIN / -1) traps on x86
    // and wraps on the GPU.
    template<typename T>
    __host__ __device__ inline T div_elem(const T& lhs, const T& rhs)
    {
        if constexpr (std::is_integral<T>::value)
            return rhs == T{0} ? T{0} : static_cast<T>(lhs / rhs);
        else
            return lhs / rhs;
    }
}
