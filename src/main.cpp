#include "mat.cuh"
#include "tensor.cuh"
#include "mat_utils.h"

int main()
{
    om::Device dv("cuda:0");
    om::Tensor<float16_t> tf1({10, 10}, dv);
    tf1.fill(10.0f);

    om::Tensor<float16_t> res = tf1 * float16_t(2.0f);
    om::print(res);
}