#pragma once

namespace om {
    
    // Runtime device dispatch
    template<typename T>
    inline void _fill(MatView<T> mat, T value, DEVICE_TYPE device) {
        switch (device) {
            case DEVICE_TYPE::CPU:  fill_dispatch<DEVICE_TYPE::CPU, T>::exec(mat, value); break;
            case DEVICE_TYPE::CUDA: fill_dispatch<DEVICE_TYPE::CUDA, T>::exec(mat, value); break;
        }
    }

}
