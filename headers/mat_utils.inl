#include "tensor.cuh"

#include <iostream>
#include <vector>
#include <iomanip>

namespace om 
{    
    template <typename T>
    void print(const Tensor<T>& tensor) {
        size_t total = tensor.size();
        std::vector<T> host_data(total);
        tensor.copyToHost(host_data.data());
    
        std::cout << "[Tensor of rank " << tensor.rank() << " | shape: ";
        for (size_t i = 0; i < tensor.rank(); ++i)
            std::cout << tensor.shape()[i] << (i + 1 < tensor.rank() ? " x " : "");
        std::cout << "]\n";
    
        if (tensor.rank() == 1) {
            for (size_t i = 0; i < tensor.shape()[0]; ++i)
                std::cout << host_data[i] << " ";
            std::cout << "\n";
        }
        else if (tensor.rank() == 2) {
            size_t rows = tensor.shape()[0];
            size_t cols = tensor.shape()[1];
            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c)
                    std::cout << std::setw(8) << host_data[r * cols + c] << " ";
                std::cout << "\n";
            }
        }
        else if (tensor.rank() == 3) {
            auto s = tensor.shape();
            for (size_t d = 0; d < s[0]; ++d) {
                std::cout << "[depth " << d << "]\n";
                for (size_t r = 0; r < s[1]; ++r) {
                    for (size_t c = 0; c < s[2]; ++c) {
                        size_t idx = d * tensor.stride()[0] + r * tensor.stride()[1] + c * tensor.stride()[2];
                        std::cout << std::setw(8) << host_data[idx] << " ";
                    }
                    std::cout << "\n";
                }
            }
        }
        else if (tensor.rank() == 4) {
            auto s = tensor.shape();
            for (size_t n = 0; n < s[0]; ++n) {
                for (size_t c = 0; c < s[1]; ++c) {
                    std::cout << "[n=" << n << ", c=" << c << "]\n";
                    for (size_t h = 0; h < s[2]; ++h) {
                        for (size_t w = 0; w < s[3]; ++w) {
                            size_t idx = n * tensor.stride()[0] + c * tensor.stride()[1] +
                                         h * tensor.stride()[2] + w * tensor.stride()[3];
                            std::cout << std::setw(6) << host_data[idx] << " ";
                        }
                        std::cout << "\n";
                    }
                }
            }
        }
        else {
            std::cout << "[flat data]: ";
            for (size_t i = 0; i < total; ++i)
                std::cout << host_data[i] << " ";
            std::cout << "\n";
        }
    }    
}
    
