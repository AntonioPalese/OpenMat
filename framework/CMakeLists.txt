cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(OpenMat LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(HEADERS
    "headers/mat_utils.h" 
    "headers/mat.cuh" 
    "headers/utils.h" 
    "headers/mat.inl" 
    )

set(SRC
    "src/mat_utils.cpp" 
    "src/utils.cpp" 
    )

find_package(CUDA 11.5 REQUIRED)

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
    )