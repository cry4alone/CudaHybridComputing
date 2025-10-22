cmake_minimum_required(VERSION 3.24)
project(work3_core LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)


file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(work3_core SHARED ${SRC})
add_library(work3::core ALIAS work3_core)

set_target_properties(work3_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(work3_core PRIVATE
    cuda_std_20
)

target_compile_options(work3_core PRIVATE
    -Wall -O3 -Wno-deprecated-gpu-targets
)

# Исправляем путь с include на includes
target_include_directories(work3_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/
)
