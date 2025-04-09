// #pragma once
// #include "tensor_view.cuh"
// #include "tensor.cuh"
// #include "cuda_defines.cuh"
// #include <vector>
// #include <cstring>

// namespace om {

// template<typename T>
// class DeviceTensorView {
// public:
//     DeviceTensorView(const T* data, const size_t* h_shape, const size_t* h_stride, size_t rank)
//         : m_Data(const_cast<T*>(data)), m_Rank(rank)
//     {
//         // Allocate and copy shape
//         CUDA_CALL(cudaMalloc(&m_Shape, sizeof(size_t) * rank));
//         CUDA_CALL(cudaMemcpy(m_Shape, h_shape, sizeof(size_t) * rank, cudaMemcpyHostToDevice));

//         // Allocate and copy stride
//         CUDA_CALL(cudaMalloc(&m_Stride, sizeof(size_t) * rank));
//         CUDA_CALL(cudaMemcpy(m_Stride, h_stride, sizeof(size_t) * rank, cudaMemcpyHostToDevice));
//     }

//     // No copy constructor
//     DeviceTensorView(const DeviceTensorView&) = delete;
//     DeviceTensorView& operator=(const DeviceTensorView&) = delete;

//     // Move constructor
//     DeviceTensorView(DeviceTensorView&& other) noexcept {
//         *this = std::move(other);
//     }

//     DeviceTensorView& operator=(DeviceTensorView&& other) noexcept {
//         if (this != &other) {
//             free_device_metadata();
//             m_Data   = other.m_Data;
//             m_Shape  = other.m_Shape;
//             m_Stride = other.m_Stride;
//             m_Rank   = other.m_Rank;
//             other.m_Shape = nullptr;
//             other.m_Stride = nullptr;
//         }
//         return *this;
//     }

//     ~DeviceTensorView() {
//         free_device_metadata();
//     }

//     // Get raw device-compatible view
//     __host__ __device__
//     TensorView<T> view() const {
//         return TensorView<T>{m_Data, m_Shape, m_Stride, m_Rank};
//     }

// private:
//     T* m_Data = nullptr;
//     size_t* m_Shape = nullptr;
//     size_t* m_Stride = nullptr;
//     size_t m_Rank = 0;

//     void free_device_metadata() {
//         if (m_Shape) cudaFree(m_Shape);
//         if (m_Stride) cudaFree(m_Stride);
//     }
// };

// } // namespace om
