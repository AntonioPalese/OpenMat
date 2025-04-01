#include "mat_utils.h"

om::Device::Device(int id, const std::string &str, om::DEVICE_TYPE type): m_Id(id), m_Str(str), m_Dt(type){}

om::Device::Device(int id, om::DEVICE_TYPE dt) : m_Id(id), m_Dt(dt)
{
    switch (dt)
    {
        case DEVICE_TYPE::CUDA: m_Str = utils::format("cuda:{}", id); break;
        case DEVICE_TYPE::CPU: m_Str = utils::format("cpu:{}", id); break;
    default:
        throw std::logic_error("Unavailable device type!");
    }
}

om::DEVICE_TYPE om::str_to_enum(const std::string &src)
{
    if(src == "cpu") return DEVICE_TYPE::CPU;
    if(src == "cuda") return DEVICE_TYPE::CUDA;
    throw std::logic_error("Unavailable device type!");
}
