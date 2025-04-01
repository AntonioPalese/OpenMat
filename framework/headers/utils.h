#pragma once

#include <string>
#include <sstream>
#include <stdexcept>

namespace utils {

    void replace_first_placeholder(std::string& out, const std::string& value);
    template<typename T>
    std::string to_string(const T& val) {
        std::ostringstream oss;
        oss << val;
        return oss.str();
    }    

    template<typename... Args>
    std::string format(const std::string& fmt, Args&&... args) {
        std::string result = fmt;
        (replace_first_placeholder(result, to_string(std::forward<Args>(args))), ...);

        // sanity check: if any '{}' left, too few arguments
        if (result.find("{}") != std::string::npos)
            throw std::runtime_error("Not enough arguments for format string");

        return result;
    }
}
