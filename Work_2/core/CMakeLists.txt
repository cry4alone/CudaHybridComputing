cmake_minimum_required(VERSION 3.24)
project(work2_core LANGUAGES CUDA)

find_package(CUDAToolkit REQUIRED)

# ВАРИАНТ А: Закомментируем include и зададим архитектуру вручную
# include(../../cmake/configs/cuda_arch.cmake)
# Вместо этого устанавливаем архитектуру явно. Узнайте свою архитектуру (например, 75 для Turing, 86 для Ampere).
set(CMAKE_CUDA_ARCHITECTURES 75) # ЗАМЕНИТЕ НА СВОЮ АРХИТЕКТУРУ!

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)


file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(work2_core SHARED ${SRC})
add_library(work2::core ALIAS work2_core)

set_target_properties(work2_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(work2_core PRIVATE
    cuda_std_20
)

target_compile_options(work2_core PRIVATE
    -Wall -O3 -Wno-deprecated-gpu-targets
)

# Исправляем путь с include на includes
target_include_directories(work2_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/
)
