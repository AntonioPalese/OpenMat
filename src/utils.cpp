#include "utils.h"

void utils::replace_first_placeholder(std::string &out, const std::string &value)
{
    size_t pos = out.find("{}");
    if (pos == std::string::npos)
        throw std::runtime_error("Too many arguments for format string");
    out.replace(pos, 2, value);
}

std::vector<std::string> utils::split(const std::string &str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

int utils::str_to_int(const std::string &str) { return std::stoi(str); }
int utils::char_to_int(char c){return c - '0';}
