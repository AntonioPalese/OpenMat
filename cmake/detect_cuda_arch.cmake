execute_process(
    COMMAND nvcc --list-gpus
    OUTPUT_VARIABLE GPU_INFO
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REGEX MATCH "sm_([0-9]+)" _match "${GPU_INFO}")
if(CMAKE_MATCH_1)
    set(DETECTED_ARCH "${CMAKE_MATCH_1}")
    message(STATUS "Detected CUDA architecture: ${DETECTED_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_ARCH}" CACHE STRING "Auto-detected arch" FORCE)
else()
    message(WARNING "Could not detect CUDA architecture. Defaulting to 50.")
    set(CMAKE_CUDA_ARCHITECTURES 50 CACHE STRING "Default fallback arch" FORCE)
endif()