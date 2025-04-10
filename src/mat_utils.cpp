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

om::Device::Device(const char *str) : m_Str(str)
{
    std::regex pattern("^\\w+:\\d+$");
    if (!std::regex_match(m_Str, pattern))
        throw std::runtime_error(utils::format("Invalid device : '{}'", m_Str));

    const auto splitted = utils::split(m_Str, ':');
    m_Id = utils::str_to_int(splitted.back());
    m_Dt = str_to_enum(splitted.front());
}

om::DEVICE_TYPE om::str_to_enum(const std::string &src)
{
    if(src == "cpu") return DEVICE_TYPE::CPU;
    if(src == "cuda") return DEVICE_TYPE::CUDA;
    throw std::logic_error("Unavailable device type!");
}
