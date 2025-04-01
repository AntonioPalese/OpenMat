#include "utils.h"
#include <iostream>
#include <string>
#include <regex>
#include <exception>

namespace om
{
    enum class DEVICE_TYPE
    {
        CUDA,
        CPU
    };

    DEVICE_TYPE str_to_enum(const std::string& src);

    struct Device
    {
        Device() = default;
        Device(int id, const std::string& str, DEVICE_TYPE type);
        Device(int id, DEVICE_TYPE dt);
        Device(const char* str) : m_Str(str)
        {
            std::regex pattern("^\\w+:\\d+$");
            if (!std::regex_match(m_Str, pattern))
                throw std::runtime_error(utils::format("Invalid device : '{}'", m_Str));

            const auto splitted = utils::split(m_Str, ':');
            m_Id = utils::str_to_int(splitted.back());
            m_Dt = str_to_enum(splitted.front());
        }

        int m_Id;
        std::string m_Str;
        DEVICE_TYPE m_Dt;
    };
};