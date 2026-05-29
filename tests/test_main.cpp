#include "minidl/framework.h"
#include <iostream>

using namespace minidl;

int main() {
    Tensor a = Tensor::fromVector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, DeviceType::CPU, true);
    Tensor b = Tensor::fromVector({4.0f, 3.0f, 2.0f, 1.0f}, {2, 2}, DeviceType::CPU, true);

    Tensor c = a.matmul(b);
    if (c.at({0, 0}) != 8.0f) {
        std::cerr << "Matrix multiply failed" << std::endl;
        return 1;
    }

    Tensor x = Tensor::randn({2, 2}, DeviceType::CPU, true);
    Tensor y = x.relu();
    Tensor loss = y.sum(1);
    loss.backward();
    if (x.grad().numel() != x.numel()) {
        std::cerr << "Gradient shape mismatch" << std::endl;
        return 1;
    }

    Tensor scalar = Tensor::ones({}, DeviceType::CPU, true);
    scalar.backward();
    if (scalar.grad().at({}) != 1.0f) {
        std::cerr << "Scalar backward failed" << std::endl;
        return 1;
    }

    std::cout << "All unit tests passed." << std::endl;
    return 0;
}
