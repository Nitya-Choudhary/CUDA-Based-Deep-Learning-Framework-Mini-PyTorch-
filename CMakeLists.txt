cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mini_dl_framework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})

set(SOURCE_FILES
    tensor/tensor.cpp
    autograd/autograd.cpp
    nn/modules.cpp
    optim/optim.cpp
    datasets/mnist.cpp
    benchmarks/benchmark.cpp
    examples/mnist_train.cpp
)

set(CUDA_SOURCES
    cuda/cuda_kernels.cu
)

add_library(minidl STATIC
    tensor/tensor.cpp
    autograd/autograd.cpp
    nn/modules.cpp
    optim/optim.cpp
    datasets/mnist.cpp
    cuda/cuda_kernels.cu
)

target_include_directories(minidl PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})

target_compile_features(minidl PUBLIC cxx_std_17)
set_target_properties(minidl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(mnist_train examples/mnist_train.cpp)
target_link_libraries(mnist_train PRIVATE minidl)

add_executable(benchmark_runner benchmarks/benchmark.cpp)
target_link_libraries(benchmark_runner PRIVATE minidl)

add_executable(test_minidl tests/test_main.cpp)
target_link_libraries(test_minidl PRIVATE minidl)

enable_testing()
add_test(NAME TensorTests COMMAND test_minidl)
