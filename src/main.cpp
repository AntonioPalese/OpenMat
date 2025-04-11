#include "mat.cuh"
#include "tensor.cuh"
#include "mat_utils.h"

int main()
{

    float16_t a(10.5f);
    float16_t b(10.5f);

    float16_t c = a + b;


    om::Device dv("cuda:0");
    om::Tensor<float16_t> tf1({10, 10}, dv);
    tf1.fill(10.0f);

    om::Tensor<float16_t> tf2({10, 10}, dv);
    tf2.fill(10.5f);

    om::Tensor<float16_t> res = tf1 * tf2;
    om::print(res);
}