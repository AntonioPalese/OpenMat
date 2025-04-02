#pragma once

#include "utils.h"
#include <iostream>
#include <string>
#include <regex>
#include <exception>

namespace om
{
    template<typename T> class Mat;
};

namespace om
{
    enum class DEVICE_TYPE
    {
        CUDA,
        CPU
    };

    DEVICE_TYPE str_to_enum(const std::string& src);

    template<typename T>
    void print(const Mat<T>& mat);    

    struct Device
    {
        Device() = default;
        Device(int id, const std::string& str, DEVICE_TYPE type);
        Device(int id, DEVICE_TYPE dt);
        Device(const char* str);

        int m_Id;
        std::string m_Str;
        DEVICE_TYPE m_Dt;
    };
};

#include "mat_utils.inl"
    