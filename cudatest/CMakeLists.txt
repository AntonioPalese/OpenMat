cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cudatest LANGUAGES CXX CUDA)

set(SRC 
    "src/matmul.cu" 
    "src/mat.cpp" 
    "src/matops.cu"
    "src/cudautils.cu"
    )
set(HEADERS 
    "headers/matmul.h" 
    "headers/mat.h" 
    "headers/matops.h"
    "headers/cudautils.h"
    )

find_package(CUDA 9.0 REQUIRED)

add_library(${PROJECT_NAME} SHARED ${SRC} ${HEADERS})

target_include_directories(${PROJECT_NAME} PUBLIC headers/)

add_executable(${PROJECT_NAME}_app src/main.cpp)

target_link_libraries(${PROJECT_NAME}_app ${PROJECT_NAME})

