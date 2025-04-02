#include "mat.cuh"

namespace om 
{
    template<typename T>
    void print(const Mat<T>& mat) {
        int rows = mat.rows();
        int cols = mat.cols();
        int count = rows * cols;
    
        std::vector<T> host_data(count);
        mat.copyToHost(host_data.data());  // assume mat is in GPU memory
    
        std::cout << "[ " << rows << "x" << cols << " matrix ]\n";
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                std::cout << host_data[r * cols + c] << "\t";
            }
            std::cout << "\n";
        }
    }
}