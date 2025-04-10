#pragma once
#include "mat_view.cuh"

namespace om {

    template<typename T>
    void fill_cpu(MatView<T> mat, T value) {
        for (int r = 0; r < mat.rows; ++r) {
            for (int c = 0; c < mat.cols; ++c) {
                mat(r, c) = value;
            }
        }
    }

}
