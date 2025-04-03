#pragma once
#include "mat_view.cuh"
#include "mat.cuh"
#include <type_traits>
#include <stdexcept>

namespace om 
{
    template<typename T>
    void add_cpu(MatView<T> lhs, MatView<T> rhs, MatView<T> dst) {
        static_assert(std::is_arithmetic_v<T>, "add_cpu requires an arithmetic type");

        if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) {
            throw std::runtime_error("Matrix dimensions must match for addition");
        }

        for (int r = 0; r < lhs.rows; ++r) {
            for (int c = 0; c < lhs.cols; ++c) {
                dst(r, c) = lhs(r, c) + rhs(r, c);
            }
        }
    }
}
