cmake_minimum_required(VERSION 3.25)
project(gpu_programming CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(gpu_programming main.cu)

set_target_properties(gpu_programming PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
