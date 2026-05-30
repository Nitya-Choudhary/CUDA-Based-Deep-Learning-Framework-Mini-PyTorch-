#pragma once

#include <memory>
#include <vector>
#include "tensor/tensor.h"

namespace minidl {

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& x) = 0;
    virtual std::vector<Tensor*> parameters() { return {}; }
    virtual void zero_grad() {}
};

class Linear : public Module {
public:
    Linear(int input_size, int output_size, DeviceType device = DeviceType::CPU, bool bias = true);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor*> parameters() override;
    void zero_grad() override;
    Tensor weight;
    Tensor bias;
    bool has_bias;
};

class ReLU : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

class Tanh : public Module {
public:
    Tensor forward(const Tensor& x) override;
};

class Softmax : public Module {
public:
    explicit Softmax(int axis = 1);
    Tensor forward(const Tensor& x) override;
private:
    int axis_;
};

class Sequential : public Module {
public:
    Sequential() = default;
    void add(std::shared_ptr<Module> module);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor*> parameters() override;
    void zero_grad() override;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace minidl
