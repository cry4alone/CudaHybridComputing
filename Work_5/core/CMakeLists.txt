cmake_minimum_required(VERSION 3.24)
project(work4_core LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)


set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_view.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/matmul_shmem_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/matmul_wmma_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/launch/matmul_shmem_launch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/launch/matmul_wmma_launch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vecred_kernels.cu
)

add_library(work4_core SHARED ${SRC})
add_library(work4::core ALIAS work4_core)

set_target_properties(work4_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(work4_core PRIVATE
    cuda_std_20
)

target_compile_options(work4_core PRIVATE
    -Wall -O3 -Wno-deprecated-gpu-targets
)

# Исправляем путь с include на includes
target_include_directories(work4_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/
)

# Enable WMMA path by default
target_compile_definitions(work4_core PUBLIC WORK4_USE_WMMA=1)
