#pragma once
// The DLPack ABI, as much of it as OpenMat uses.
//
// DLPack is the in-memory tensor exchange standard used by PyTorch, NumPy,
// CuPy, JAX and the Python Array API (https://github.com/dmlc/dlpack,
// Apache License 2.0). These declarations reproduce the layout of the
// unversioned `DLManagedTensor` from dlpack.h v0.8, which is what a
// "dltensor" capsule carries; field order and types are the ABI and must not
// change. The versioned `DLManagedTensorVersioned` (DLPack 1.x) is not used:
// every consumer still accepts the legacy capsule.
#include <cstdint>

extern "C" {

typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
} DLDeviceType;

typedef struct {
    DLDeviceType device_type;
    int32_t device_id;
} DLDevice;

typedef enum {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLOpaqueHandle = 3,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLBool = 6,
} DLDataTypeCode;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;       // in elements; NULL means compact row-major
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

} // extern "C"
