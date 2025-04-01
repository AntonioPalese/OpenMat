#include "utils.h"

void utils::replace_first_placeholder(std::string &out, const std::string &value)
{
    size_t pos = out.find("{}");
    if (pos == std::string::npos)
        throw std::runtime_error("Too many arguments for format string");
    out.replace(pos, 2, value);
}
