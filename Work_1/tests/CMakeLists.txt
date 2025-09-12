cmake_minimum_required(VERSION 3.24)
project(work1_tests LANGUAGES CXX)

set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)

# Закомментируем эти строки, так как файлов нет
# include(../../cmake/deps/gtest.cmake)
# include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work1_tests ${SRC})

target_compile_features(work1_tests PRIVATE
    cxx_std_20
)

target_compile_options(work1_tests PRIVATE
    -Wall -O3
)

target_link_libraries(work1_tests PRIVATE
    work1::core
    CUDA::cudart
    GTest::gtest_main # Будет найден через find_package в главном CMakeLists
    Eigen3::Eigen     # Будет найден через find_package в главном CMakeLists
)

enable_testing()
gtest_discover_tests(work1_tests)