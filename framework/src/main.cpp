#include "mat.cuh"

int main()
{
    om::Mat<float> m(100, 100, "cuda:0");
    m.fill(0.4f);

    om::print(m);
}