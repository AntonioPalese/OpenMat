#pragma once
#include "mat_view.cuh"

namespace om {

    template<typename T>
    __global__ void fill_kernel(MatView<T> mat, T value);

    template<typename T>
    void launch_fill(MatView<T> mat, T value);
}
