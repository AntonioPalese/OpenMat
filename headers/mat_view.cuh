#pragma once

namespace om {
    template<typename T>
    struct MatView {
        T* data;
        int rows;
        int cols;

        __host__ __device__
        inline T& operator()(int r, int c) {
            return data[r * cols + c];
        }

        __host__ __device__
        inline const T& operator()(int r, int c) const {
            return data[r * cols + c];
        }

        __host__ __device__
        inline int index(int r, int c) const {
            return r * cols + c;
        }
    };
}
