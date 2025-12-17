cmake_minimum_required(VERSION 3.24)
project(work5_core LANGUAGES CUDA)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vecred_nobr_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vecred_br_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/launch/vecred_nobr_launch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/launch/vecred_br_launch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/launch/vecred_reduce_utils.cu
)

add_library(work5_core SHARED ${SRC})
add_library(work5::core ALIAS work5_core)

set_target_properties(work5_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(work5_core PRIVATE
    cuda_std_20
)

target_compile_options(work5_core PRIVATE
    -Wall -O3 -Wno-deprecated-gpu-targets
)

# Исправляем путь с include на includes
target_include_directories(work5_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/
)

# Enable WMMA path by default
target_compile_definitions(work5_core PUBLIC WORK5_USE_WMMA=1)
