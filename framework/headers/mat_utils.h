#include "utils.h"
#include <iostream>
#include <string>



enum class DEVICE_TYPE
{
    CUDA,
    CPU
};

struct Device
{
    Device() = default;
    Device(int id, const std::string& str, DEVICE_TYPE type);
    Device(int id, DEVICE_TYPE dt);
    Device(const std::string& str) : m_Str(str)
    {
        const auto splitted = utils::split(m_Str, ':');
        m_Id = utils::str_to_int(splitted.front());
        //m_Id = utils::str_to_int(splitted.front());
    }

    int m_Id;
    std::string m_Str;
    DEVICE_TYPE m_Dt;
};