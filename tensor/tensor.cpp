#include "tensor/tensor.h"
#include "cuda/cuda_utils.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <limits>
#include <algorithm>

namespace minidl {

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<int>& shape, DeviceType device, bool requires_grad)
    : device_(device), shape_(shape), requires_grad_(requires_grad) {
    numel_ = compute_numel(shape_);
    compute_strides();
    allocate_storage();
    if (requires_grad_) {
        grad_ = std::make_shared<Tensor>(zeros(shape_, device_, false));
    }
}

Tensor::Tensor(const std::vector<int>& shape, float value, DeviceType device, bool requires_grad)
    : Tensor(shape, device, requires_grad) {
    if (numel_ == 0) return;
    if (device_ == DeviceType::CPU) {
        for (size_t i = 0; i < numel_; ++i) {
            data_.get()[i] = value;
        }
    } else {
        Tensor cpu_value = Tensor::zeros(shape_, DeviceType::CPU);
        for (size_t i = 0; i < numel_; ++i) {
            cpu_value.data_.get()[i] = value;
        }
        cuda_ops::check_cuda(cudaMemcpy(data_.get(), cpu_value.data_.get(), numel_ * sizeof(float), cudaMemcpyHostToDevice));
    }
}

Tensor Tensor::zeros(const std::vector<int>& shape, DeviceType device, bool requires_grad) {
    return Tensor(shape, device, requires_grad);
}

Tensor Tensor::ones(const std::vector<int>& shape, DeviceType device, bool requires_grad) {
    Tensor result(shape, device, requires_grad);
    if (result.numel_ == 0) return result;
    if (device == DeviceType::CPU) {
        for (size_t i = 0; i < result.numel_; ++i) {
            result.data_.get()[i] = 1.0f;
        }
    } else {
        Tensor cpu_data = Tensor::zeros(shape, DeviceType::CPU);
        for (size_t i = 0; i < cpu_data.numel_; ++i) {
            cpu_data.data_.get()[i] = 1.0f;
        }
        cuda_ops::check_cuda(cudaMemcpy(result.data_.get(), cpu_data.data_.get(), result.numel_ * sizeof(float), cudaMemcpyHostToDevice));
    }
    return result;
}

Tensor Tensor::randn(const std::vector<int>& shape, DeviceType device, bool requires_grad) {
    Tensor result(shape, DeviceType::CPU, requires_grad);
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    for (size_t i = 0; i < result.numel_; ++i) {
        result.data_.get()[i] = distribution(generator);
    }
    if (device == DeviceType::CUDA) {
        result = result.to(DeviceType::CUDA);
        result.requires_grad_ = requires_grad;
        if (requires_grad_) {
            result.grad_ = std::make_shared<Tensor>(zeros(shape, device, false));
        }
    }
    return result;
}

Tensor Tensor::fromVector(const std::vector<float>& data, const std::vector<int>& shape, DeviceType device, bool requires_grad) {
    Tensor result(shape, device, requires_grad);
    assert(result.numel_ == data.size());
    if (device == DeviceType::CPU) {
        std::memcpy(result.data_.get(), data.data(), data.size() * sizeof(float));
    } else {
        Tensor cpu_data = Tensor::zeros(shape, DeviceType::CPU);
        std::memcpy(cpu_data.data_.get(), data.data(), data.size() * sizeof(float));
        cuda_ops::check_cuda(cudaMemcpy(result.data_.get(), cpu_data.data_.get(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    return result;
}

void Tensor::allocate_storage() {
    if (numel_ == 0) {
        data_.reset();
        return;
    }
    if (device_ == DeviceType::CPU) {
        data_ = std::shared_ptr<float>(new float[numel_], std::default_delete<float[]>());
        std::memset(data_.get(), 0, numel_ * sizeof(float));
    } else {
        float* device_ptr = nullptr;
        cuda_ops::check_cuda(cudaMalloc(&device_ptr, numel_ * sizeof(float)));
        cuda_ops::check_cuda(cudaMemset(device_ptr, 0, numel_ * sizeof(float)));
        data_ = std::shared_ptr<float>(device_ptr, [](float* ptr) { cudaFree(ptr); });
    }
}

void Tensor::compute_strides() {
    strides_.assign(shape_.size(), 1);
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::compute_numel(const std::vector<int>& shape) {
    size_t total = 1;
    for (int dim : shape) {
        total *= dim;
    }
    return total;
}

Tensor Tensor::to(DeviceType device) const {
    Tensor result(shape_, device, requires_grad_);
    if (device == device_) {
        if (numel_ > 0) {
            if (device == DeviceType::CPU) {
                std::memcpy(result.data_.get(), data_.get(), numel_ * sizeof(float));
            } else {
                cuda_ops::check_cuda(cudaMemcpy(result.data_.get(), data_.get(), numel_ * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }
        return result;
    }
    if (device_ == DeviceType::CPU && device == DeviceType::CUDA) {
        cuda_ops::check_cuda(cudaMemcpy(result.data_.get(), data_.get(), numel_ * sizeof(float), cudaMemcpyHostToDevice));
    } else if (device_ == DeviceType::CUDA && device == DeviceType::CPU) {
        cuda_ops::check_cuda(cudaMemcpy(result.data_.get(), data_.get(), numel_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return result;
}

void Tensor::copy_from(const Tensor& source) {
    assert(shape_ == source.shape_);
    if (device_ == source.device_) {
        if (device_ == DeviceType::CPU) {
            std::memcpy(data_.get(), source.data_.get(), numel_ * sizeof(float));
        } else {
            cuda_ops::check_cuda(cudaMemcpy(data_.get(), source.data_.get(), numel_ * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    } else {
        Tensor temp = source.to(device_);
        std::memcpy(data_.get(), temp.data_.get(), numel_ * sizeof(float));
    }
}

void Tensor::reshape(const std::vector<int>& shape) {
    assert(compute_numel(shape) == numel_);
    shape_ = shape;
    compute_strides();
}

size_t Tensor::flatten_index(const std::vector<int>& indices) const {
    assert(indices.size() == shape_.size());
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        assert(indices[i] >= 0 && indices[i] < shape_[i]);
        offset += indices[i] * strides_[i];
    }
    return offset;
}

float Tensor::at(const std::vector<int>& indices) const {
    size_t idx = flatten_index(indices);
    if (device_ == DeviceType::CPU) {
        return data_.get()[idx];
    }
    float value = 0.0f;
    cuda_ops::check_cuda(cudaMemcpy(&value, data_.get() + idx, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

void Tensor::set(const std::vector<int>& indices, float value) {
    size_t idx = flatten_index(indices);
    if (device_ == DeviceType::CPU) {
        data_.get()[idx] = value;
    } else {
        cuda_ops::check_cuda(cudaMemcpy(data_.get() + idx, &value, sizeof(float), cudaMemcpyHostToDevice));
    }
}

std::string Tensor::to_string() const {
    std::ostringstream buffer;
    buffer << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        buffer << shape_[i];
        if (i + 1 < shape_.size()) buffer << ", ";
    }
    buffer << "], device=" << (device_ == DeviceType::CUDA ? "CUDA" : "CPU") << ", values=[";
    Tensor host_view = (device_ == DeviceType::CUDA ? to(DeviceType::CPU) : *this);
    size_t display = std::min<size_t>(host_view.numel_, 8);
    for (size_t i = 0; i < display; ++i) {
        buffer << std::fixed << std::setprecision(4) << host_view.data_.get()[i];
        if (i + 1 < display) buffer << ", ";
    }
    if (host_view.numel_ > display) buffer << ", ...";
    buffer << "]";
    return buffer.str();
}

void Tensor::save(const std::string& path) const {
    Tensor host_view = (device_ == DeviceType::CUDA ? to(DeviceType::CPU) : *this);
    std::ofstream output(path, std::ios::binary);
    output << static_cast<int>(shape_.size()) << " ";
    for (int dim : shape_) {
        output << dim << " ";
    }
    output.write(reinterpret_cast<const char*>(host_view.data_.get()), numel_ * sizeof(float));
}

Tensor Tensor::load(const std::string& path, DeviceType device, bool requires_grad) {
    std::ifstream input(path, std::ios::binary);
    int rank = 0;
    input >> rank;
    std::vector<int> shape(rank);
    for (int i = 0; i < rank; ++i) {
        input >> shape[i];
    }
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    size_t total = compute_numel(shape);
    std::vector<float> buffer(total);
    input.read(reinterpret_cast<char*>(buffer.data()), total * sizeof(float));
    return Tensor::fromVector(buffer, shape, device, requires_grad);
}

Tensor& Tensor::grad() {
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(zeros(shape_, device_, false));
    }
    return *grad_;
}

const Tensor& Tensor::grad() const {
    assert(grad_);
    return *grad_;
}

void Tensor::zero_grad() {
    if (!requires_grad_) return;
    grad_ = std::make_shared<Tensor>(zeros(shape_, device_, false));
}

void Tensor::build_topo(std::vector<Tensor>& topo, std::unordered_set<GradOp*>& visited) const {
    if (!grad_op_ || !visited.insert(grad_op_.get()).second) {
        return;
    }
    for (auto& parent : grad_op_->parents) {
        parent.build_topo(topo, visited);
    }
    topo.push_back(*this);
}

void Tensor::add_grad(const Tensor& grad) {
    Tensor actual_grad = grad.device_ == device_ ? grad : grad.to(device_);
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(actual_grad);
        return;
    }
    if (grad_->numel_ == 0) {
        *grad_ = actual_grad;
        return;
    }
    assert(grad_->numel_ == actual_grad.numel_);
    if (device_ == DeviceType::CPU) {
        for (size_t i = 0; i < numel_; ++i) {
            grad_->data_.get()[i] += actual_grad.data_.get()[i];
        }
    } else {
        Tensor grad_cpu = grad_->to(DeviceType::CPU);
        Tensor actual_cpu = actual_grad.to(DeviceType::CPU);
        for (size_t i = 0; i < numel_; ++i) {
            grad_cpu.data_.get()[i] += actual_cpu.data_.get()[i];
        }
        *grad_ = grad_cpu.to(device_);
    }
}

void Tensor::backward() {
    if (!requires_grad_) return;
    if (!grad_ || grad_->numel_ == 0) {
        grad_ = std::make_shared<Tensor>(ones(shape_, device_, false));
    }
    std::vector<Tensor> topo;
    std::unordered_set<GradOp*> visited;
    build_topo(topo, visited);
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if (it->grad_op_) {
            it->grad_op_->backward();
        }
    }
}

Tensor Tensor::clone() const {
    Tensor result(shape_, device_, requires_grad_);
    if (numel_ > 0) {
        if (device_ == DeviceType::CPU) {
            std::memcpy(result.data_.get(), data_.get(), numel_ * sizeof(float));
        } else {
            cuda_ops::check_cuda(cudaMemcpy(result.data_.get(), data_.get(), numel_ * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    return result;
}

static void copy_elementwise(const Tensor& a, const Tensor& b, Tensor& out, const std::function<float(float, float)>& op) {
    assert(a.numel_ == b.numel_);
    for (size_t i = 0; i < a.numel_; ++i) {
        out.data_.get()[i] = op(a.data_.get()[i], b.data_.get()[i]);
    }
}

Tensor Tensor::operator+(const Tensor& other) const {
    assert(shape_ == other.shape_);
    assert(device_ == other.device_);
    Tensor out(shape_, device_, requires_grad_ || other.requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::add(*this, other, out);
    } else {
        copy_elementwise(*this, other, out, [](float x, float y) { return x + y; });
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this, other};
        out.grad_op_->backward = [lhs = *this, rhs = other, out]() mutable {
            if (lhs.requires_grad_) {
                lhs.add_grad(out.grad());
            }
            if (rhs.requires_grad_) {
                rhs.add_grad(out.grad());
            }
        };
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    assert(shape_ == other.shape_);
    assert(device_ == other.device_);
    Tensor out(shape_, device_, requires_grad_ || other.requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::subtract(*this, other, out);
    } else {
        copy_elementwise(*this, other, out, [](float x, float y) { return x - y; });
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this, other};
        out.grad_op_->backward = [lhs = *this, rhs = other, out]() mutable {
            if (lhs.requires_grad_) {
                lhs.add_grad(out.grad());
            }
            if (rhs.requires_grad_) {
                rhs.add_grad(out.grad() * -1.0f);
            }
        };
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    assert(shape_ == other.shape_);
    assert(device_ == other.device_);
    Tensor out(shape_, device_, requires_grad_ || other.requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::multiply(*this, other, out);
    } else {
        copy_elementwise(*this, other, out, [](float x, float y) { return x * y; });
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this, other};
        out.grad_op_->backward = [lhs = *this, rhs = other, out]() mutable {
            if (lhs.requires_grad_) {
                lhs.add_grad(out.grad() * rhs);
            }
            if (rhs.requires_grad_) {
                rhs.add_grad(out.grad() * lhs);
            }
        };
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const {
    assert(shape_ == other.shape_);
    assert(device_ == other.device_);
    Tensor out(shape_, device_, requires_grad_ || other.requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::divide(*this, other, out);
    } else {
        copy_elementwise(*this, other, out, [](float x, float y) { return x / y; });
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this, other};
        out.grad_op_->backward = [lhs = *this, rhs = other, out]() mutable {
            if (lhs.requires_grad_) {
                lhs.add_grad(out.grad() / rhs);
            }
            if (rhs.requires_grad_) {
                Tensor numerator = lhs * out.grad() * -1.0f;
                Tensor denominator = rhs * rhs;
                rhs.add_grad(numerator / denominator);
            }
        };
    }
    return out;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor out(shape_, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::add_scalar(*this, out, scalar);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = data_.get()[i] + scalar;
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this};
        out.grad_op_->backward = [lhs = *this, out]() mutable {
            if (lhs.requires_grad_) {
                lhs.add_grad(out.grad());
            }
        };
    }
    return out;
}

Tensor Tensor::operator-(float scalar) const {
    return *this + (-scalar);
}

Tensor Tensor::operator*(float scalar) const {
    Tensor out(shape_, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::multiply_scalar(*this, out, scalar);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = data_.get()[i] * scalar;
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this};
        out.grad_op_->backward = [lhs = *this, scalar, out]() mutable {
            if (lhs.requires_grad_) {
                lhs.add_grad(out.grad() * scalar);
            }
        };
    }
    return out;
}

Tensor Tensor::operator/(float scalar) const {
    assert(scalar != 0.0f);
    return *this * (1.0f / scalar);
}

Tensor Tensor::matmul(const Tensor& other) const {
    assert(shape_.size() == 2 && other.shape_.size() == 2);
    assert(shape_[1] == other.shape_[0]);
    assert(device_ == other.device_);
    Tensor out({shape_[0], other.shape_[1]}, device_, requires_grad_ || other.requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::matmul(*this, other, out);
    } else {
        int M = shape_[0];
        int K = shape_[1];
        int N = other.shape_[1];
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += data_.get()[i * K + k] * other.data_.get()[k * N + j];
                }
                out.data_.get()[i * N + j] = sum;
            }
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this, other};
        out.grad_op_->backward = [lhs = *this, rhs = other, out]() mutable {
            if (lhs.requires_grad_) {
                lhs.add_grad(out.grad().matmul(rhs.transpose()));
            }
            if (rhs.requires_grad_) {
                rhs.add_grad(lhs.transpose().matmul(out.grad()));
            }
        };
    }
    return out;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    assert(shape_.size() == 2);
    std::vector<int> result_shape = {shape_[1], shape_[0]};
    Tensor out(result_shape, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::transpose(*this, out);
    } else {
        for (int i = 0; i < shape_[0]; ++i) {
            for (int j = 0; j < shape_[1]; ++j) {
                out.data_.get()[j * shape_[0] + i] = data_.get()[i * shape_[1] + j];
            }
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this};
        out.grad_op_->backward = [lhs = *this, out]() mutable {
            if (lhs.requires_grad_) {
                lhs.add_grad(out.grad().transpose());
            }
        };
    }
    return out;
}

Tensor Tensor::sum(int axis) const {
    if (axis == -1) axis = static_cast<int>(shape_.size()) - 1;
    assert(axis >= 0 && axis < static_cast<int>(shape_.size()));
    if (shape_.size() == 1) {
        float accumulator = 0.0f;
        if (device_ == DeviceType::CPU) {
            for (size_t i = 0; i < numel_; ++i) {
                accumulator += data_.get()[i];
            }
        } else {
            Tensor host = to(DeviceType::CPU);
            for (size_t i = 0; i < numel_; ++i) {
                accumulator += host.data_.get()[i];
            }
        }
        Tensor result({}, device_, requires_grad_);
        if (device_ == DeviceType::CPU) {
            result.data_.get()[0] = accumulator;
        } else {
            cuda_ops::check_cuda(cudaMemcpy(result.data_.get(), &accumulator, sizeof(float), cudaMemcpyHostToDevice));
        }
        if (result.requires_grad_) {
            result.grad_op_ = std::make_shared<GradOp>();
            result.grad_op_->parents = {*this};
            result.grad_op_->backward = [lhs = *this, result]() mutable {
                if (lhs.requires_grad_) {
                    lhs.add_grad(Tensor::ones(lhs.shape_, lhs.device_, false) * result.grad());
                }
            };
        }
        return result;
    }
    assert(shape_.size() == 2);
    std::vector<int> out_shape = shape_;
    out_shape[axis] = 1;
    Tensor out(out_shape, device_, requires_grad_);
    if (device_ == DeviceType::CUDA && axis == 1) {
        cuda_ops::reduce_sum_last_dim(*this, out);
    } else {
        int rows = shape_[0];
        int cols = shape_[1];
        if (axis == 1) {
            for (int i = 0; i < rows; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < cols; ++j) {
                    sum += data_.get()[i * cols + j];
                }
                out.data_.get()[i] = sum;
            }
        } else {
            for (int j = 0; j < cols; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < rows; ++i) {
                    sum += data_.get()[i * cols + j];
                }
                out.data_.get()[j] = sum;
            }
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this};
        out.grad_op_->backward = [lhs = *this, out, axis]() mutable {
            if (lhs.requires_grad_) {
                Tensor grad_output = out.grad();
                Tensor grads = Tensor::zeros(lhs.shape_, lhs.device_, false);
                if (axis == 1) {
                    for (int i = 0; i < lhs.shape_[0]; ++i) {
                        float value = grad_output.data_.get()[i];
                        for (int j = 0; j < lhs.shape_[1]; ++j) {
                            grads.data_.get()[i * lhs.shape_[1] + j] = value;
                        }
                    }
                } else {
                    for (int j = 0; j < lhs.shape_[1]; ++j) {
                        float value = grad_output.data_.get()[j];
                        for (int i = 0; i < lhs.shape_[0]; ++i) {
                            grads.data_.get()[i * lhs.shape_[1] + j] = value;
                        }
                    }
                }
                lhs.add_grad(grads);
            }
        };
    }
    return out;
}

Tensor Tensor::mean(int axis) const {
    Tensor sum_tensor = sum(axis);
    float divisor = static_cast<float>(shape_[axis]);
    return sum_tensor / divisor;
}

Tensor Tensor::relu() const {
    Tensor out(shape_, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::relu(*this, out);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = std::max(0.0f, data_.get()[i]);
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this};
        out.grad_op_->backward = [lhs = *this, out]() mutable {
            if (lhs.requires_grad_) {
                Tensor grad_output = out.grad();
                Tensor grads = Tensor::zeros(lhs.shape_, lhs.device_, false);
                for (size_t i = 0; i < lhs.numel_; ++i) {
                    grads.data_.get()[i] = (lhs.data_.get()[i] > 0.0f ? grad_output.data_.get()[i] : 0.0f);
                }
                lhs.add_grad(grads);
            }
        };
    }
    return out;
}

Tensor Tensor::sigmoid() const {
    Tensor out(shape_, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::sigmoid(*this, out);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = 1.0f / (1.0f + std::exp(-data_.get()[i]));
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this};
        out.grad_op_->backward = [lhs = *this, out]() mutable {
            if (lhs.requires_grad_) {
                Tensor grad_output = out.grad();
                Tensor grads = Tensor::zeros(lhs.shape_, lhs.device_, false);
                for (size_t i = 0; i < lhs.numel_; ++i) {
                    float s = out.data_.get()[i];
                    grads.data_.get()[i] = grad_output.data_.get()[i] * s * (1.0f - s);
                }
                lhs.add_grad(grads);
            }
        };
    }
    return out;
}

Tensor Tensor::tanh() const {
    Tensor out(shape_, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::tanh(*this, out);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = std::tanh(data_.get()[i]);
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this};
        out.grad_op_->backward = [lhs = *this, out]() mutable {
            if (lhs.requires_grad_) {
                Tensor grad_output = out.grad();
                Tensor grads = Tensor::zeros(lhs.shape_, lhs.device_, false);
                for (size_t i = 0; i < lhs.numel_; ++i) {
                    float t = out.data_.get()[i];
                    grads.data_.get()[i] = grad_output.data_.get()[i] * (1.0f - t * t);
                }
                lhs.add_grad(grads);
            }
        };
    }
    return out;
}

Tensor Tensor::log() const {
    Tensor out(shape_, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::log(*this, out);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = std::log(std::max(data_.get()[i], 1e-20f));
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this};
        out.grad_op_->backward = [lhs = *this, out]() mutable {
            if (lhs.requires_grad_) {
                Tensor grad_output = out.grad();
                Tensor grads = Tensor::zeros(lhs.shape_, lhs.device_, false);
                for (size_t i = 0; i < lhs.numel_; ++i) {
                    grads.data_.get()[i] = grad_output.data_.get()[i] / std::max(lhs.data_.get()[i], 1e-20f);
                }
                lhs.add_grad(grads);
            }
        };
    }
    return out;
}

Tensor Tensor::softmax(int axis) const {
    assert(shape_.size() == 2);
    Tensor out(shape_, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::softmax(*this, out, axis);
    } else {
        int rows = shape_[0];
        int cols = shape_[1];
        for (int i = 0; i < rows; ++i) {
            float max_val = -FLT_MAX;
            for (int j = 0; j < cols; ++j) {
                max_val = std::max(max_val, data_.get()[i * cols + j]);
            }
            float sum = 0.0f;
            for (int j = 0; j < cols; ++j) {
                out.data_.get()[i * cols + j] = std::exp(data_.get()[i * cols + j] - max_val);
                sum += out.data_.get()[i * cols + j];
            }
            for (int j = 0; j < cols; ++j) {
                out.data_.get()[i * cols + j] /= std::max(sum, 1e-8f);
            }
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this};
        out.grad_op_->backward = [lhs = *this, out]() mutable {
            if (lhs.requires_grad_) {
                Tensor grad_output = out.grad();
                Tensor grads = Tensor::zeros(lhs.shape_, lhs.device_, false);
                int rows = lhs.shape_[0];
                int cols = lhs.shape_[1];
                for (int i = 0; i < rows; ++i) {
                    float dot = 0.0f;
                    for (int j = 0; j < cols; ++j) {
                        dot += out.data_.get()[i * cols + j] * grad_output.data_.get()[i * cols + j];
                    }
                    for (int j = 0; j < cols; ++j) {
                        grads.data_.get()[i * cols + j] = out.data_.get()[i * cols + j] * (grad_output.data_.get()[i * cols + j] - dot);
                    }
                }
                lhs.add_grad(grads);
            }
        };
    }
    return out;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    assert(shape_ == other.shape_);
    *this = *this + other;
    return *this;
}

Tensor& Tensor::operator*=(float scalar) {
    *this = *this * scalar;
    return *this;
}

} // namespace minidl
