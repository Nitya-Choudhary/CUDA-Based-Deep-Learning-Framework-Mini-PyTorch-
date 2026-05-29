#pragma once

#include <cstdint>
#include <vector>
#include "tensor/tensor.h"

namespace minidl {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor*> parameters);
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad();
protected:
    std::vector<Tensor*> params_;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> parameters, float lr = 0.01f, float momentum = 0.9f);
    void step() override;
private:
    float lr_;
    float momentum_;
    std::vector<Tensor> velocity_;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    void step() override;
private:
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int64_t timestep_;
    std::vector<Tensor> m1_;
    std::vector<Tensor> m2_;
};

} // namespace minidl
