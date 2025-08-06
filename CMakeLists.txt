cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(OpenMat LANGUAGES CXX CUDA)

# Compile with  : cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -B build

# Load dynamic CUDA arch detection
#include("${CMAKE_SOURCE_DIR}/cmake/detect_cuda_arch.cmake")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

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

set(CMAKE_CUDA_ARCHITECTURES 61)

message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ----------------------------------------
# CPU files
# ----------------------------------------
file(GLOB HEADERS "headers/*.h" "headers/*.cuh" "headers/*.inl" "headers/ops/cpu/*.h")

file(GLOB SRC "src/*.cpp" "src/ops/cpu/*.cpp")
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
