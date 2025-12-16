cmake_minimum_required(VERSION 3.24)
project(work4 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_BENCH "Enable benchmarks" ON)

include(FetchContent)
include(GoogleTest)

if(ENABLE_TESTS OR ENABLE_BENCH)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(googletest)
endif()

if(ENABLE_BENCH)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3 
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable gtest for benchmark")
set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "Don't use bundled gtest")

FetchContent_MakeAvailable(googlebenchmark)
endif()

find_package(Eigen3 3.4.0 REQUIRED)

add_subdirectory(core)
set_target_properties(work4_core PROPERTIES CUDA_ARCHITECTURES 75)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_BENCH)
    add_subdirectory(benchmark)
endif()