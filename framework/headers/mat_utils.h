#include "utils.h"

enum class DEVICE_TYPE
{
    CUDA,
    CPU
};

struct Device
{
    Device(int id, DEVICE_TYPE dt);

    int m_Id;
    std::string m_Str;
    DEVICE_TYPE m_Dt;
};