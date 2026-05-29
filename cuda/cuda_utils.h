#pragma once

#include <cuda_runtime.h>
#include "tensor/tensor.h"

namespace minidl {
namespace cuda_ops {

void check_cuda(cudaError_t result);

void add(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream = 0);
void subtract(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream = 0);
void multiply(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream = 0);
void divide(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream = 0);
void add_scalar(const Tensor& in, Tensor& out, float scalar, cudaStream_t stream = 0);
void multiply_scalar(const Tensor& in, Tensor& out, float scalar, cudaStream_t stream = 0);
void divide_scalar(const Tensor& in, Tensor& out, float scalar, cudaStream_t stream = 0);
void matmul(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream = 0);
void transpose(const Tensor& in, Tensor& out, cudaStream_t stream = 0);
void reduce_sum_last_dim(const Tensor& in, Tensor& out, cudaStream_t stream = 0);
void relu(const Tensor& in, Tensor& out, cudaStream_t stream = 0);
void sigmoid(const Tensor& in, Tensor& out, cudaStream_t stream = 0);
void tanh(const Tensor& in, Tensor& out, cudaStream_t stream = 0);
void log(const Tensor& in, Tensor& out, cudaStream_t stream = 0);
void softmax(const Tensor& in, Tensor& out, int axis = 1, cudaStream_t stream = 0);

void sgd_update(Tensor& param, const Tensor& grad, float lr, float momentum, Tensor& velocity, cudaStream_t stream = 0);
void adam_update(Tensor& param, const Tensor& grad, float lr, float beta1, float beta2, float epsilon, Tensor& m1, Tensor& m2, int64_t timestep, cudaStream_t stream = 0);

} // namespace cuda_ops
} // namespace minidl
