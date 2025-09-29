cmake_minimum_required(VERSION 3.24)
project(work2_benchmarks LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(work2_benchmarks unit_tests.cpp)

set(CMAKE_CUDA_ARCHITECTURES 50)


set_target_properties(work2_benchmarks PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(work2_benchmarks PRIVATE
    cxx_std_20
)

target_compile_options(work2_benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work2_benchmarks PRIVATE
    work1::core
    CUDA::cudart
    GTest::gtest_main # Будет найден через find_package в главном CMakeLists
    Eigen3::Eigen     # Будет найден через find_package в главном CMakeLists
)

enable_testing()
gtest_discover_tests(work2_benchmarks)