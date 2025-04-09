#include "mat.cuh"
#include "tensor.cuh"
#include "mat_utils.h"

int main()
{
    om::Device dv("cuda:0");
    om::Tensor<float> tf({2, 3, 10, 10}, dv);
    tf.fill(10.0f);
    om::print(tf);
}