cmake_minimum_required(VERSION 3.24)
project(work1_core LANGUAGES CUDA)

# ВАРИАНТ А: Закомментируем include и зададим архитектуру вручную
# include(../../cmake/configs/cuda_arch.cmake)
# Вместо этого устанавливаем архитектуру явно. Узнайте свою архитектуру (например, 75 для Turing, 86 для Ampere).
set(CMAKE_CUDA_ARCHITECTURES 86) # ЗАМЕНИТЕ НА СВОЮ АРХИТЕКТУРУ!

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(work1_core SHARED ${SRC})
add_library(work1::core ALIAS work1_core)

set_target_properties(work1_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

# Если используете set_target_minimum_cuda_arch, то раскомментируйте include и эту строку
# set_target_minimum_cuda_arch(work1_core 75)

target_compile_features(work1_core PRIVATE
    cuda_std_20
)

target_compile_options(work1_core PRIVATE
    -Wall -O3 -Wno-deprecated-gpu-targets
)

# Исправляем путь с include на includes
target_include_directories(work1_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/
)

# Убираем линковку на несуществующий cudagh
# target_link_libraries(work1_core PRIVATE work1::cudagh)