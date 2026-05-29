#include "minidl/framework.h"
#include <chrono>
#include <iostream>
#include <string>

using namespace minidl;

static double elapsed_ms(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "=== Mini DL Framework Benchmark ===\n";

    Tensor a = Tensor::randn({1 << 15, 16}, DeviceType::CPU);
    Tensor b = Tensor::randn({16, 32}, DeviceType::CPU);

    auto start = std::chrono::steady_clock::now();
    Tensor c = a.matmul(b);
    auto end = std::chrono::steady_clock::now();
    std::cout << "CPU matmul: " << elapsed_ms(start, end) << " ms" << std::endl;

    int cuda_count = 0;
    if (cudaGetDeviceCount(&cuda_count) == cudaSuccess && cuda_count > 0) {
        Tensor ac = a.to(DeviceType::CUDA);
        Tensor bc = b.to(DeviceType::CUDA);
        cudaDeviceSynchronize();

        auto start_cuda = std::chrono::steady_clock::now();
        Tensor cc = ac.matmul(bc);
        cudaDeviceSynchronize();
        auto end_cuda = std::chrono::steady_clock::now();
        std::cout << "CUDA matmul: " << elapsed_ms(start_cuda, end_cuda) << " ms" << std::endl;
    } else {
        std::cout << "CUDA device unavailable; skipping GPU benchmarks." << std::endl;
    }

    std::cout << "Simple elementwise and activation tests...\n";
    Tensor x = Tensor::randn({1024, 1024}, DeviceType::CPU);
    Tensor y = Tensor::randn({1024, 1024}, DeviceType::CPU);
    auto begin = std::chrono::steady_clock::now();
    Tensor z = x + y;
    Tensor relu = z.relu();
    auto finish = std::chrono::steady_clock::now();
    std::cout << "CPU elementwise+ReLU: " << elapsed_ms(begin, finish) << " ms" << std::endl;

    std::cout << "Benchmark report:\n";
    std::cout << "  CPU matrix multiply throughput: " << (double)a.numel() * b.shape()[1] / elapsed_ms(start, end) << " elements/ms\n";
    std::cout << "  CPU tensor memory footprint: " << (a.numel() + b.numel() + c.numel()) * sizeof(float) / 1024.0 / 1024.0 << " MB\n";

    std::cout << "Benchmark complete." << std::endl;
    return 0;
}
