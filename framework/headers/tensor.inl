
#include "tensor.cuh"

template<typename value_type>
om::Tensor<value_type>::Tensor(const std::vector<size_t>& shape, const Device & dv): m_Shape(shape), m_Device(device) 
{
    compute_strides();
    total_size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    allocate_memory();  // cudaMalloc or malloc depending on device
}