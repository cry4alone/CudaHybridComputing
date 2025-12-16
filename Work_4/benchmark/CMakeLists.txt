cmake_minimum_required(VERSION 3.24)
project(work4_benchmarks LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

find_package(CUDAToolkit REQUIRED)
# find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work4_benchmarks src/perf_benchmark.cu)

target_compile_features(work4_benchmarks PRIVATE
    cxx_std_20
)

target_compile_options(work4_benchmarks PRIVATE
    -Wall -O3
)

target_link_libraries(work4_benchmarks PRIVATE
    work4::core
    CUDA::cudart
    benchmark::benchmark 
    Eigen3::Eigen        
)