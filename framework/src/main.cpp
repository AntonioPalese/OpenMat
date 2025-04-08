#include "mat.cuh"
#include "tensor.cuh"
#include "mat_utils.h"

int main()
{
    om::Device dv("cuda:0");
    om::Mat<float> m1(10, 10, dv);
    m1.fill(10.0f);

    om::Mat<float> m2(10, 10, dv);
    m2.fill(2.0f);

    om::Mat res = m1/m2;
    std::cout << "matrix element type : " << res.dtype() << "\n";
    om::print(res);

    om::Tensor<float> tf({2, 2}, dv);
}