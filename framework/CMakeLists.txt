cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
project(OpenMat LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 86)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

file(GLOB HEADERS "headers/*.h" "headers/*.cuh" "headers/*.inl")

file(GLOB SRC "src/*.cpp")
list(REMOVE_ITEM SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

file(GLOB CUDA_SRC "src/*.cu")
list(APPEND SRC ${CUDA_SRC})

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