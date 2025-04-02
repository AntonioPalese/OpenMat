#pragma once

namespace om {

    template<>
    inline void _fill<DEVICE_TYPE::CPU, float>(MatView<float> mat, float value) {
        fill_cpu(mat, value);
    }

    template<>
    inline void _fill<DEVICE_TYPE::CUDA, float>(MatView<float> mat, float value) {
        launch_fill(mat, value);
    }

    // Runtime device dispatch
    template<typename T>
    inline void _fill(MatView<T> mat, T value, DEVICE_TYPE device) {
        switch (device) {
            case DEVICE_TYPE::CPU: _fill<DEVICE_TYPE::CPU>(mat, value); break;
            case DEVICE_TYPE::CUDA: _fill<DEVICE_TYPE::CUDA>(mat, value); break;
        }
    }

}
