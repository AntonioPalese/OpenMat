#include "mat.cuh"

int main()
{
    om::Mat<double> m(2, 2, Device(0, DEVICE_TYPE::CUDA));
}