cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

# Must be set BEFORE project()/enable_language(CUDA): enabling CUDA defines
# CMAKE_CUDA_ARCHITECTURES with CMake's own default, so this guard would never
# fire if placed any later. "native" targets the GPU in this machine
# (CMake >= 3.24); override with -DCMAKE_CUDA_ARCHITECTURES=<sm> to cross-compile.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(OpenMat LANGUAGES CXX CUDA)

# Compile with  : cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build

# Load dynamic CUDA arch detection
#include("${CMAKE_SOURCE_DIR}/cmake/detect_cuda_arch.cmake")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

if(DEFINED ENV{CMAKE_LIBRARY_PATH})
    message(STATUS "Custom flag (CMAKE_LIBRARY_PATH) is set: $ENV{CMAKE_LIBRARY_PATH}")
else()
    message(WARNING "CMAKE_LIBRARY_PATH is not set in the environment.")
endif()

string(REPLACE ":" ";" CUDA_LIB_PATH_LIST "$ENV{CMAKE_LIBRARY_PATH}")

foreach(path IN LISTS CUDA_LIB_PATH_LIST)
    message(STATUS "CUDA LIB path: ${path}")
endforeach()


cmake_policy(SET CMP0104 NEW)

message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()


# ----------------------------------------
# Debug launch checking (see headers/cuda_defines.cuh)
#
# OM_DEBUG_SYNC forces a cudaStreamSynchronize() after every kernel launch, so
# an out-of-bounds access is reported at the launching call site instead of at
# some later, unrelated synchronization point. It serializes streams, so it is
# off by default and can also be switched on per run with OPENMAT_DEBUG_SYNC=1
# without rebuilding. OM_NO_DEBUG_SYNC compiles the machinery out entirely.
# ----------------------------------------
option(OM_DEBUG_SYNC    "Synchronize after every kernel launch by default" OFF)
option(OM_NO_DEBUG_SYNC "Compile out the forced post-launch synchronization" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ----------------------------------------
# CPU files
# ----------------------------------------
file(GLOB HEADERS "headers/*.h" "headers/*.cuh" "headers/*.inl" "headers/ops/*.h" "headers/ops/cpu/*.h")

file(GLOB SRC "src/*.cpp" "src/ops/cpu/*.cpp" "src/python/*.cpp")
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

# ----------------------------------------
# CUDA files
# ----------------------------------------
file(GLOB CUDA_HEADERS "headers/ops/kernels/*.cuh")
list(APPEND HEADERS ${CUDA_HEADERS})
file(GLOB CUDA_TYPE_TRAITS_H "headers/type_traits/*.cuh")
list(APPEND HEADERS ${CUDA_TYPE_TRAITS_H})

file(GLOB CUDA_SRC "src/ops/kernels/*.cu")
list(APPEND SRC ${CUDA_SRC})
file(GLOB CUDA_TYPE_TRAITS_SRC "src/type_traits/*.cu")
list(APPEND SRC ${CUDA_TYPE_TRAITS_SRC})

set_source_files_properties(${CUDA_SRC} PROPERTIES LANGUAGE CUDA)

# ----------------------------------------
# Shared library target
# ----------------------------------------
add_library(${PROJECT_NAME} SHARED ${SRC} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "OpenMat"
    SUFFIX ".so"      # optional, default on Linux
    PREFIX ""          # remove lib
)

target_include_directories(${PROJECT_NAME} PUBLIC headers/)

# matmul_cpu.h (headers/ops/cpu/matmul_cpu.h) parallelizes its outer loop with
# #pragma omp. PUBLIC so every consumer that includes tensor.cuh and therefore
# re-instantiates the header-only template (tests, src/main.cpp, the Python
# capi TU) compiles that instantiation with the same -fopenmp flags — mixing a
# -fopenmp and a non--fopenmp instantiation of the same weak symbol across TUs
# is an ODR violation, not just a missed optimization.
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

if(OM_DEBUG_SYNC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE OM_DEBUG_SYNC=1)
endif()
if(OM_NO_DEBUG_SYNC)
    target_compile_definitions(${PROJECT_NAME} PRIVATE OM_NO_DEBUG_SYNC=1)
endif()


# ----------------------------------------
# Main executable
# ----------------------------------------
add_executable(${PROJECT_NAME}_app src/main.cpp)

target_link_libraries(${PROJECT_NAME}_app 
            ${PROJECT_NAME}
            cuda
            cudart
    )

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

foreach(path IN LISTS CUDA_LIB_PATH_LIST)
        target_link_directories(${PROJECT_NAME}  PUBLIC ${path} )
endforeach()

enable_testing()
add_subdirectory(tests)
