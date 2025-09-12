cmake_minimum_required(VERSION 3.24)
project(work1 LANGUAGES CXX CUDA)

set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_BENCH "Enable benchmarks" ON)

# --- Настройка зависимостей ---
include(FetchContent)
include(GoogleTest) # Для gtest_discover_tests

# 1. Сначала загружаем GoogleTest
if(ENABLE_TESTS OR ENABLE_BENCH)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0 # Используем более новую версию
)
FetchContent_MakeAvailable(googletest)
endif()

# 2. Затем настраиваем и загружаем Google Benchmark
if(ENABLE_BENCH)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3 # Более новая стабильная версия
)

# КРИТИЧЕСКИ ВАЖНО: Отключаем встроенные зависимости Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable gtest for benchmark")
set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "Don't use bundled gtest")

FetchContent_MakeAvailable(googlebenchmark)
endif()

# 3. Ищем Eigen
find_package(Eigen3 3.4.0 REQUIRED)

add_subdirectory(core)

# if(ENABLE_TESTS)
#     add_subdirectory(tests)
# endif()

# if(ENABLE_BENCH)
#     add_subdirectory(benchmark)
# endif()