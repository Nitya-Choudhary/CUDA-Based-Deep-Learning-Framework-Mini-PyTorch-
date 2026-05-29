#include "optim/optim.h"
#include "cuda/cuda_utils.h"
#include <cassert>
#include <cmath>

namespace minidl {

Optimizer::Optimizer(std::vector<Tensor*> parameters)
    : params_(std::move(parameters)) {}

void Optimizer::zero_grad() {
    for (Tensor* param : params_) {
        if (param && param->requires_grad()) {
            param->zero_grad();
        }
    }
}

SGD::SGD(std::vector<Tensor*> parameters, float lr, float momentum)
    : Optimizer(std::move(parameters)), lr_(lr), momentum_(momentum) {
    velocity_.reserve(params_.size());
    for (auto* param : params_) {
        velocity_.emplace_back(Tensor::zeros(param->shape(), param->device(), false));
    }
}

void SGD::step() {
    assert(params_.size() == velocity_.size());
    for (size_t idx = 0; idx < params_.size(); ++idx) {
        Tensor* param = params_[idx];
        Tensor& velocity = velocity_[idx];
        assert(param->grad().numel() == param->numel());
        if (param->device() == DeviceType::CUDA) {
            cuda_ops::sgd_update(*param, param->grad(), lr_, momentum_, velocity);
        } else {
            float* p = param->data();
            float* g = param->grad().data();
            float* v = velocity.data();
            size_t count = param->numel();
            for (size_t i = 0; i < count; ++i) {
                float delta = momentum_ * v[i] - lr_ * g[i];
                v[i] = delta;
                p[i] += delta;
            }
        }
    }
}

Adam::Adam(std::vector<Tensor*> parameters, float lr, float beta1, float beta2, float epsilon)
    : Optimizer(std::move(parameters)), lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), timestep_(0) {
    m1_.reserve(params_.size());
    m2_.reserve(params_.size());
    for (auto* param : params_) {
        m1_.emplace_back(Tensor::zeros(param->shape(), param->device(), false));
        m2_.emplace_back(Tensor::zeros(param->shape(), param->device(), false));
    }
}

void Adam::step() {
    assert(params_.size() == m1_.size());
    assert(params_.size() == m2_.size());
    timestep_ += 1;
    for (size_t idx = 0; idx < params_.size(); ++idx) {
        Tensor* param = params_[idx];
        Tensor& m1 = m1_[idx];
        Tensor& m2 = m2_[idx];
        assert(param->grad().numel() == param->numel());
        if (param->device() == DeviceType::CUDA) {
            cuda_ops::adam_update(*param, param->grad(), lr_, beta1_, beta2_, epsilon_, m1, m2, timestep_);
        } else {
            float* p = param->data();
            float* g = param->grad().data();
            float* m1_data = m1.data();
            float* m2_data = m2.data();
            size_t count = param->numel();
            float correction1 = 1.0f - std::pow(beta1_, static_cast<float>(timestep_));
            float correction2 = 1.0f - std::pow(beta2_, static_cast<float>(timestep_));
            for (size_t i = 0; i < count; ++i) {
                m1_data[i] = beta1_ * m1_data[i] + (1.0f - beta1_) * g[i];
                m2_data[i] = beta2_ * m2_data[i] + (1.0f - beta2_) * g[i] * g[i];
                float m1_hat = m1_data[i] / correction1;
                float m2_hat = m2_data[i] / correction2;
                p[i] -= lr_ * m1_hat / (std::sqrt(m2_hat) + epsilon_);
            }
        }
    }
}

} // namespace minidl
