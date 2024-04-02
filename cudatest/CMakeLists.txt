cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cudatest LANGUAGES CXX CUDA)


add_executable(${PROJECT_NAME} src/main.cpp src/matmul.cu headers/matmul.h)
target_include_directories(${PROJECT_NAME} PRIVATE headers/)
