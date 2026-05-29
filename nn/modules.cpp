#include "nn/modules.h"
#include "cuda/cuda_utils.h"
#include <cassert>
#include <vector>

namespace minidl {

Linear::Linear(int input_size, int output_size, DeviceType device, bool bias)
    : weight({input_size, output_size}, device, true), bias({1, output_size}, device, true), has_bias(bias) {
    weight = Tensor::randn({input_size, output_size}, device, true) * 0.01f;
    if (has_bias) {
        bias = Tensor::zeros({1, output_size}, device, true);
    }
}

Tensor Linear::forward(const Tensor& x) {
    Tensor output = x.matmul(weight);
    if (!has_bias) {
        return output;
    }

    const int batch = x.shape()[0];
    const int features = bias.shape()[1];
    Tensor bias_cpu = bias.device() == DeviceType::CUDA ? bias.to(DeviceType::CPU) : bias;
    std::vector<float> expanded_data(batch * features);

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < features; ++j) {
            expanded_data[i * features + j] = bias_cpu.data()[j];
        }
    }

    Tensor expanded_bias = Tensor::fromVector(expanded_data, {batch, features}, bias.device(), false);
    return output + expanded_bias;
}

std::vector<Tensor*> Linear::parameters() {
    std::vector<Tensor*> params;
    params.push_back(&weight);
    if (has_bias) params.push_back(&bias);
    return params;
}

void Linear::zero_grad() {
    if (weight.requires_grad()) {
        weight.zero_grad();
    }
    if (has_bias && bias.requires_grad()) {
        bias.zero_grad();
    }
}

Tensor ReLU::forward(const Tensor& x) {
    return x.relu();
}

Tensor Sigmoid::forward(const Tensor& x) {
    return x.sigmoid();
}

Tensor Tanh::forward(const Tensor& x) {
    return x.tanh();
}

Softmax::Softmax(int axis) : axis_(axis) {}

Tensor Softmax::forward(const Tensor& x) {
    return x.softmax(axis_);
}

void Sequential::add(std::shared_ptr<Module> module) {
    modules_.push_back(std::move(module));
}

Tensor Sequential::forward(const Tensor& x) {
    Tensor output = x;
    for (auto& module : modules_) {
        output = module->forward(output);
    }
    return output;
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> params;
    for (auto& module : modules_) {
        auto module_params = module->parameters();
        params.insert(params.end(), module_params.begin(), module_params.end());
    }
    return params;
}

void Sequential::zero_grad() {
    for (auto& module : modules_) {
        module->zero_grad();
    }
}

} // namespace minidl
