cmake_minimum_required(VERSION 3.24)
project(work2_benchmarks LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

# Закомментируем эти строки, так как файлов нет
# include(../../cmake/deps/gbench.cmake)
# include(../../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)
# find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work2_benchmarks src/perf_benchmark.cu)

target_compile_features(work2_benchmarks PRIVATE
    cxx_std_20
)

target_compile_options(work2_benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work2_benchmarks PRIVATE
    work2::core
    # work1::cuda_timer # Убираем, так как этой библиотеки нет
    CUDA::cudart
    benchmark::benchmark # Будет найден через find_package/FetchContent в главном CMakeLists
    Eigen3::Eigen        # Будет найден через find_package/FetchContent в главном CMakeLists
)