namespace om {
    
    // Runtime device dispatch
    template<typename T>
    inline void _fill(TensorView<T> tensor, T value, DEVICE_TYPE device) {
        switch (device) {
            case DEVICE_TYPE::CPU:  fill_dispatch<DEVICE_TYPE::CPU, T>::exec(tensor, value); break;
            case DEVICE_TYPE::CUDA: fill_dispatch<DEVICE_TYPE::CUDA, T>::exec(tensor, value); break;
        }
    }

    DEFINE_DEVICE_DISPATCH_BINARY_INL(add)
    DEFINE_DEVICE_DISPATCH_BINARY_INL(sub)
    DEFINE_DEVICE_DISPATCH_BINARY_INL(mul)
    DEFINE_DEVICE_DISPATCH_BINARY_INL(div)

    DEFINE_DEVICE_DISPATCH_UNARY_INL(add_k)
    DEFINE_DEVICE_DISPATCH_UNARY_INL(sub_k)
    DEFINE_DEVICE_DISPATCH_UNARY_INL(mul_k)
    DEFINE_DEVICE_DISPATCH_UNARY_INL(div_k)
}