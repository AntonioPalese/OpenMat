#pragma once
#include "tensor_view.cuh"
#include <stdexcept>
#include <limits>

namespace om
{
    template<typename T>
    T reduce_sum_cpu(const TensorView<const T> src) {
        size_t n = src.size();

        // A single accumulator forces a loop-carried FP dependency chain
        // (one addition per FP-add latency, not per throughput) that the
        // compiler cannot auto-vectorize without -ffast-math. Splitting the
        // sum across independent partial accumulators breaks that chain -
        // the compiler can run/pipeline them in parallel - and merging pairs
        // of similar-magnitude partials at the end is also more accurate
        // than one long straight-line accumulation.
        constexpr size_t LANES = 8;
        T acc[LANES];
        for (size_t j = 0; j < LANES; ++j) acc[j] = static_cast<T>(0);

        size_t i = 0;
        const size_t limit = n - (n % LANES);
        for (; i < limit; i += LANES) {
            for (size_t j = 0; j < LANES; ++j) acc[j] = acc[j] + src[i + j];
        }

        T result = static_cast<T>(0);
        for (size_t j = 0; j < LANES; ++j) result = result + acc[j];
        for (; i < n; ++i) result = result + src[i];
        return result;
    }

    template<typename T>
    T reduce_min_cpu(const TensorView<const T> src) {
        if (src.size() == 0) throw std::invalid_argument("reduce_min: empty tensor");
        T acc = src[0];
        size_t n = src.size();
        for (size_t i = 1; i < n; ++i) if (src[i] < acc) acc = src[i];
        return acc;
    }

    template<typename T>
    T reduce_max_cpu(const TensorView<const T> src) {
        if (src.size() == 0) throw std::invalid_argument("reduce_max: empty tensor");
        T acc = src[0];
        size_t n = src.size();
        for (size_t i = 1; i < n; ++i) if (src[i] > acc) acc = src[i];
        return acc;
    }
}
