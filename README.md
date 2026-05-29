# Mini DL Framework

A modular C++ / CUDA deep learning framework inspired by PyTorch. This repository includes a complete tensor engine, CUDA-accelerated kernels, autograd computation graph, neural network modules, optimizers, dataset loaders, benchmarking tools, and example training pipelines.

## Project Structure

- `tensor/` - Core `Tensor` engine, shape handling, device abstraction, CPU/GPU allocation, operators, serialization.
- `cuda/` - CUDA kernels, optimized matrix multiplication, transpose, reductions, activations, optimizer update kernels.
- `autograd/` - Automatic differentiation engine, computational graph, backward propagation, gradient accumulation.
- `nn/` - Neural network layers and model containers.
- `optim/` - SGD and Adam optimizers.
- `datasets/` - MNIST dataset loader and preprocessing utilities.
- `benchmarks/` - Benchmark harness for CPU vs CUDA performance comparisons.
- `examples/` - Training example for MNIST classification.
- `tests/` - Unit tests for tensor operations and autograd.
- `include/` - Public headers that expose the framework API.

## Requirements

- Linux with NVIDIA GPU and CUDA Toolkit
- CMake 3.18 or newer
- NVIDIA CUDA-enabled compiler

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -- -j
```

## Run

```bash
./mnist_train
./benchmark_runner
ctest --verbose
```

## CUDA Optimization Notes

- `cuda/cuda_kernels.cu` implements shared-memory tiling for matrix multiplication and transpose.
- Highly coalesced memory accesses are used in pointwise operations.
- Kernel launches are tuned for warp-aligned block sizes.
- Optimizer updates use GPU kernels for in-place parameter updates.

## Usage

```cpp
#include <minidl/framework.h>

using namespace minidl;

Tensor x = Tensor::randn({64, 784}, DeviceType::CUDA, true);
Tensor w = Tensor::randn({784, 128}, DeviceType::CUDA, true);
Tensor y = x.matmul(w).relu();
Tensor loss = y.mean();
loss.backward();
```

## Notes

- The MNIST example uses a local dataset loader. Download raw MNIST files and point the executable to the dataset folder.
- The benchmark harness reports elapsed time, throughput, and memory usage estimates.
- This project is intended as a portfolio-quality starter deep learning framework in modern C++ and CUDA.
