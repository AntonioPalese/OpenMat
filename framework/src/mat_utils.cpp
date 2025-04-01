#include "mat_utils.h"

Device::Device(int id, const std::string &str, DEVICE_TYPE type): m_Id(id), m_Str(str), m_Dt(type){}

Device::Device(int id, DEVICE_TYPE dt) : m_Id(id), m_Dt(dt)
{
    switch (dt)
    {
        case DEVICE_TYPE::CUDA: m_Str = utils::format("cuda:{}", id); break;
        case DEVICE_TYPE::CPU: m_Str = utils::format("cpu:{}", id); break;
    default:
        throw std::logic_error("Unavailable device type!");
    }
}