#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

namespace minidl {

enum class DeviceType { CPU, CUDA };

class Tensor;

struct GradOp {
    std::vector<Tensor> parents;
    std::function<void()> backward;
};

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int>& shape, DeviceType device = DeviceType::CPU, bool requires_grad = false);
    Tensor(const std::vector<int>& shape, float value, DeviceType device = DeviceType::CPU, bool requires_grad = false);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    ~Tensor() = default;

    static Tensor zeros(const std::vector<int>& shape, DeviceType device = DeviceType::CPU, bool requires_grad = false);
    static Tensor ones(const std::vector<int>& shape, DeviceType device = DeviceType::CPU, bool requires_grad = false);
    static Tensor randn(const std::vector<int>& shape, DeviceType device = DeviceType::CPU, bool requires_grad = false);
    static Tensor fromVector(const std::vector<float>& data, const std::vector<int>& shape, DeviceType device = DeviceType::CPU, bool requires_grad = false);

    Tensor to(DeviceType device) const;
    void copy_from(const Tensor& source);
    void reshape(const std::vector<int>& shape);
    float at(const std::vector<int>& indices) const;
    void set(const std::vector<int>& indices, float value);
    std::string to_string() const;
    void save(const std::string& path) const;
    static Tensor load(const std::string& path, DeviceType device = DeviceType::CPU, bool requires_grad = false);

    size_t numel() const { return numel_; }
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    DeviceType device() const { return device_; }
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool enabled) { requires_grad_ = enabled; }
    void zero_grad();
    void backward();

    Tensor clone() const;

    float* data() const { return data_.get(); }
    Tensor& grad();
    const Tensor& grad() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    Tensor matmul(const Tensor& other) const;
    Tensor transpose(int dim0 = 0, int dim1 = 1) const;
    Tensor sum(int axis = -1) const;
    Tensor mean(int axis = -1) const;
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor log() const;
    Tensor softmax(int axis = 1) const;

    Tensor &operator+=(const Tensor& other);
    Tensor &operator*=(float scalar);

private:
    void allocate_storage();
    void compute_strides();
    static size_t compute_numel(const std::vector<int>& shape);
    size_t flatten_index(const std::vector<int>& indices) const;
    void add_grad(const Tensor& grad);
    void build_topo(std::vector<Tensor>& topo, std::unordered_set<GradOp*>& visited) const;

    DeviceType device_ = DeviceType::CPU;
    std::vector<int> shape_;
    std::vector<int> strides_;
    size_t numel_ = 0;
    std::shared_ptr<float> data_;
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;
    std::shared_ptr<GradOp> grad_op_;
};

Tensor mse_loss(const Tensor& prediction, const Tensor& target);
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& target);

} // namespace minidl
