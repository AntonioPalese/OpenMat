#include "mat.cuh"
#include "mat_utils.h"

int main()
{
    om::Device dv("cuda:0");
    om::Mat<float> m1(4, 4, dv);
    m1.fill(5.0f);

    om::Mat<float> m2(4, 4, dv);
    m2.fill(2.0f);

    om::Mat res = m1.mul(m2);
    om::print(res);
}