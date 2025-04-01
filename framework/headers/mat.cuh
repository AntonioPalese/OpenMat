#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <memory>

#include "mat_utils.h"
#include "allocator.h"


namespace om
{
    template<typename _Ty>
    class Mat
    {
    public:
        using value_type = _Ty;

        Mat(int r, int c, const Device& dv = Device(0, DEVICE_TYPE::CPU));
        Mat(const Mat& rhs); // copy
        Mat(Mat&& rhs); // move

        ~Mat();

        Device device() const {return m_device;}
    private:
        int m_Rows;
        int m_Cols;
        _Ty* m_Data;
        Device m_device;

        std::unique_ptr<Allocator<_Ty>> m_Allocator;
    };
}

#include "mat.inl"
