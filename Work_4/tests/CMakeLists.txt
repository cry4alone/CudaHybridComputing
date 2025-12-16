cmake_minimum_required(VERSION 3.24)
project(work4_tests LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

find_package(CUDAToolkit REQUIRED)

set_source_files_properties(src/unit_tests.cpp PROPERTIES LANGUAGE CUDA)

add_executable(work4_tests src/unit_tests.cpp)

set(CMAKE_CUDA_ARCHITECTURES 50)


set_target_properties(work4_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(work4_tests PRIVATE
    cxx_std_20
)

target_compile_options(work4_tests PRIVATE
    -Wall -O3
)

target_link_libraries(work4_tests PRIVATE
    work4::core
    CUDA::cudart
    GTest::gtest_main 
    Eigen3::Eigen     
)

enable_testing()
gtest_discover_tests(work4_tests)