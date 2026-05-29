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

namespace minidl {

static void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<int>& shape, DeviceType device, bool requires_grad)
    : device_(device), shape_(shape), requires_grad_(requires_grad) {
    numel_ = compute_numel(shape_);
    compute_strides();
    allocate_storage();
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
        check_cuda(cudaMemcpy(data_.get(), cpu_value.data_.get(), numel_ * sizeof(float), cudaMemcpyHostToDevice));
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
        check_cuda(cudaMemcpy(result.data_.get(), cpu_data.data_.get(), result.numel_ * sizeof(float), cudaMemcpyHostToDevice));
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
        check_cuda(cudaMemcpy(result.data_.get(), cpu_data.data_.get(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
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
        check_cuda(cudaMalloc(&device_ptr, numel_ * sizeof(float)));
        check_cuda(cudaMemset(device_ptr, 0, numel_ * sizeof(float)));
        data_ = std::shared_ptr<float>(device_ptr, [](float* ptr) { cudaFree(ptr); });
    }
}

void Tensor::compute_strides() {
    strides_.assign(shape_.size(), 1);
    for (int i = (int)shape_.size() - 2; i >= 0; --i) {
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
                check_cuda(cudaMemcpy(result.data_.get(), data_.get(), numel_ * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }
        return result;
    }
    if (device_ == DeviceType::CPU && device == DeviceType::CUDA) {
        check_cuda(cudaMemcpy(result.data_.get(), data_.get(), numel_ * sizeof(float), cudaMemcpyHostToDevice));
    } else if (device_ == DeviceType::CUDA && device == DeviceType::CPU) {
        check_cuda(cudaMemcpy(result.data_.get(), data_.get(), numel_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return result;
}

void Tensor::copy_from(const Tensor& source) {
    assert(shape_ == source.shape_);
    if (device_ == source.device_) {
        if (device_ == DeviceType::CPU) {
            std::memcpy(data_.get(), source.data_.get(), numel_ * sizeof(float));
        } else {
            check_cuda(cudaMemcpy(data_.get(), source.data_.get(), numel_ * sizeof(float), cudaMemcpyDeviceToDevice));
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
    check_cuda(cudaMemcpy(&value, data_.get() + idx, sizeof(float), cudaMemcpyDeviceToHost));
    return value;
}

void Tensor::set(const std::vector<int>& indices, float value) {
    size_t idx = flatten_index(indices);
    if (device_ == DeviceType::CPU) {
        data_.get()[idx] = value;
        return;
    }
    check_cuda(cudaMemcpy(data_.get() + idx, &value, sizeof(float), cudaMemcpyHostToDevice));
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
    size_t display = std::min<size_t>(host_view.numel(), 8);
    for (size_t i = 0; i < display; ++i) {
        buffer << std::fixed << std::setprecision(4) << host_view.data_.get()[i];
        if (i + 1 < display) buffer << ", ";
    }
    if (host_view.numel() > display) buffer << ", ...";
    buffer << "])";
    return buffer.str();
}

void Tensor::save(const std::string& path) const {
    Tensor host_view = (device_ == DeviceType::CUDA ? to(DeviceType::CPU) : *this);
    std::ofstream output(path, std::ios::binary);
    output << static_cast<int>(shape_.size());
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

void Tensor::zero_grad() {
    if (!requires_grad_) return;
    grad_ = Tensor::zeros(shape_, device_, false);
}

void Tensor::build_topo(std::vector<Tensor*>& topo, std::unordered_set<Tensor*>&) {
    topo.push_back(this);
    if (!grad_op_) return;
    for (auto& parent : grad_op_->parents) {
        parent.build_topo(topo, std::unordered_set<Tensor*>());
    }
}

void Tensor::add_grad(const Tensor& grad) {
    if (grad_.numel_ == 0) {
        grad_ = grad.clone();
        return;
    }
    grad_ = grad_ + grad;
}

void Tensor::backward() {
    if (!requires_grad_) return;
    if (grad_.numel_ == 0) {
        grad_ = Tensor::ones(shape_, device_, false);
    }
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;
    build_topo(topo, visited);
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor* current = *it;
        if (current->grad_op_) {
            current->grad_op_->backward(*current);
        }
    }
}

Tensor Tensor::clone() const {
    Tensor result(shape_, device_, requires_grad_);
    if (numel_ > 0) {
        if (device_ == DeviceType::CPU) {
            std::memcpy(result.data_.get(), data_.get(), numel_ * sizeof(float));
        } else {
            check_cuda(cudaMemcpy(result.data_.get(), data_.get(), numel_ * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    assert(shape_ == other.shape_);
    Tensor out(shape_, device_, requires_grad_ || other.requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::add(*this, other, out);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = data_.get()[i] + other.data_.get()[i];
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this, other};
        out.grad_op_->backward = [this, other](Tensor& self) {
            if (this->requires_grad_) {
                this->add_grad(self.grad_);
            }
            if (other.requires_grad_) {
                other.add_grad(self.grad_);
            }
        };
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    assert(shape_ == other.shape_);
    Tensor out(shape_, device_, requires_grad_ || other.requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::subtract(*this, other, out);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = data_.get()[i] - other.data_.get()[i];
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this, other};
        out.grad_op_->backward = [this, other](Tensor& self) {
            if (this->requires_grad_) {
                this->add_grad(self.grad_);
            }
            if (other.requires_grad_) {
                other.add_grad(self.grad_ * -1.0f);
            }
        };
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    assert(shape_ == other.shape_);
    Tensor out(shape_, device_, requires_grad_ || other.requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::multiply(*this, other, out);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = data_.get()[i] * other.data_.get()[i];
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this, other};
        out.grad_op_->backward = [this, other](Tensor& self) {
            if (this->requires_grad_) {
                this->add_grad(self.grad_ * other);
            }
            if (other.requires_grad_) {
                other.add_grad(self.grad_ * *this);
            }
        };
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const {
    assert(shape_ == other.shape_);
    Tensor out(shape_, device_, requires_grad_ || other.requires_grad_);
    if (device_ == DeviceType::CUDA) {
        cuda_ops::divide(*this, other, out);
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = data_.get()[i] / other.data_.get()[i];
        }
    }
    if (out.requires_grad_) {
        out.grad_op_ = std::make_shared<GradOp>();
        out.grad_op_->parents = {*this, other};
        out.grad_op_->backward = [this, other](Tensor& self) {
            if (this->requires_grad_) {
                this->add_grad(self.grad_ / other);
            }
            if (other.requires_grad_) {
                Tensor numerator = *this * self.grad_ * -1.0f;
                Tensor denominator = other * other;
                other.add_grad(numerator / denominator);
            }
        };
    }
    return out;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor out(shape_, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        Tensor bias = Tensor::ones(shape_, device_, false) * scalar;
        out = *this + bias;
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = data_.get()[i] + scalar;
        }
    }
    return out;
}

Tensor Tensor::operator-(float scalar) const {
    return *this + (-scalar);
}

Tensor Tensor::operator*(float scalar) const {
    Tensor out(shape_, device_, requires_grad_);
    if (device_ == DeviceType::CUDA) {
        Tensor scaling = Tensor::ones(shape_, device_, false);
        for (size_t i = 0; i < numel_; ++i) {
            if (device_ == DeviceType::CPU) {
                scaling.data_.get()[i] = scalar;
            }
        }
        out = *this * scaling;
    } else {
        for (size_t i = 0; i < numel_; ++i) {
            out.data_.get()[i] = data_.get()[i] * scalar;
        }
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
    std::vector<int> shape = {shape_[0], other.shape_[1]};
    Tensor out(shape, device_, requires_grad_ || other.requires_grad_);
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
        out.grad_op_->backward = [this, other](Tensor& self) {
            if (this->requires_grad_) {
                Tensor grad_a = self.grad_.matmul(other.transpose());
                this->add_grad(grad_a);
            }
            if (other.requires_grad_) {
                Tensor grad_b = this->transpose().matmul(self.grad_);
                other.add_grad(grad_b);
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
        out.grad_op_->backward = [this](Tensor& self) {
            if (this->requires_grad_) {
                this->add_grad(self.grad_.transpose());
            }
        };
    }
    return out;
}

Tensor Tensor::sum(int axis) const {
    if (axis == -1) axis = (int)shape_.size() - 1;
    assert(axis >= 0 && axis < (int)shape_.size());
    if (shape_.size() == 1) {
        float accumulator = 0.0f;
        if (device_ == DeviceType::CPU) {
            for (size_t i = 0; i < numel_; ++i) {
                accumulator += data_.get()[i];
            }
            Tensor result({}, device_, requires_grad_);
            result.data_ = std::shared_ptr<float>(new float[1], std::default_delete<float[]>());
            result.data_.get()[0] = accumulator;
            return result;
        } else {
            Tensor host = to(DeviceType::CPU);
            for (size_t i = 0; i < numel_; ++i) {
                accumulator += host.data_.get()[i];
            }
            Tensor result({}, DeviceType::CPU, requires_grad_);
            result.data_ = std::shared_ptr<float>(new float[1], std::default_delete<float[]>());
            result.data_.get()[0] = accumulator;
            return result;
        }
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
        out.grad_op_->backward = [this, axis](Tensor& self) {
            if (this->requires_grad_) {
                Tensor expanded = self.grad_;
                if (axis == 1) {
                    expanded.reshape({shape_[0], 1});
                    Tensor grads = Tensor::zeros(shape_, device_, false);
                    for (int i = 0; i < shape_[0]; ++i) {
                        for (int j = 0; j < shape_[1]; ++j) {
                            grads.data_.get()[i * shape_[1] + j] = expanded.data_.get()[i];
                        }
                    }
                    this->add_grad(grads);
                } else {
                    expanded.reshape({1, shape_[1]});
                    Tensor grads = Tensor::zeros(shape_, device_, false);
                    for (int i = 0; i < shape_[0]; ++i) {
                        for (int j = 0; j < shape_[1]; ++j) {
                            grads.data_.get()[i * shape_[1] + j] = expanded.data_.get()[j];
                        }
                    }
                    this->add_grad(grads);
                }
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
        out.grad_op_->backward = [this](Tensor& self) {
            if (this->requires_grad_) {
                Tensor grads = Tensor::zeros(shape_, device_, false);
                for (size_t i = 0; i < numel_; ++i) {
                    grads.data_.get()[i] = (data_.get()[i] > 0.0f ? self.grad_.data_.get()[i] : 0.0f);
                }
                this->add_grad(grads);
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
        out.grad_op_->backward = [this](Tensor& self) {
            if (this->requires_grad_) {
                Tensor sigma = self;
                Tensor grads = Tensor::zeros(shape_, device_, false);
                for (size_t i = 0; i < numel_; ++i) {
                    float s = sigma.data_.get()[i];
                    grads.data_.get()[i] = self.grad_.data_.get()[i] * s * (1.0f - s);
                }
                this->add_grad(grads);
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
        out.grad_op_->backward = [this](Tensor& self) {
            if (this->requires_grad_) {
                Tensor grads = Tensor::zeros(shape_, device_, false);
                for (size_t i = 0; i < numel_; ++i) {
                    float t = self.data_.get()[i];
                    grads.data_.get()[i] = self.grad_.data_.get()[i] * (1.0f - t * t);
                }
                this->add_grad(grads);
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
        out.grad_op_->backward = [this](Tensor& self) {
            if (this->requires_grad_) {
                Tensor grads = Tensor::zeros(shape_, device_, false);
                for (size_t i = 0; i < numel_; ++i) {
                    grads.data_.get()[i] = self.grad_.data_.get()[i] / std::max(this->data_.get()[i], 1e-20f);
                }
                this->add_grad(grads);
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
        out.grad_op_->backward = [this](Tensor& self) {
            if (this->requires_grad_) {
                Tensor grad_input = Tensor::zeros(shape_, device_, false);
                int rows = shape_[0];
                int cols = shape_[1];
                for (int i = 0; i < rows; ++i) {
                    float dot = 0.0f;
                    for (int j = 0; j < cols; ++j) {
                        dot += self.data_.get()[i * cols + j] * self.grad_.data_.get()[i * cols + j];
                    }
                    for (int j = 0; j < cols; ++j) {
                        grad_input.data_.get()[i * cols + j] = self.data_.get()[i * cols + j] * (self.grad_.data_.get()[i * cols + j] - dot);
                    }
                }
                this->add_grad(grad_input);
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
