#include "mat.cuh"

int main()
{
    om::Mat<float> m(2, 2, "cuda:0");
    m.fill(3.14f);

    om::print(m);
}