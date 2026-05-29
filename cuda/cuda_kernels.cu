#include "cuda/cuda_utils.h"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <float.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace minidl {
namespace cuda_ops {

__global__ void elementwise_kernel(const float* a, const float* b, float* out, size_t n, int op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    switch (op) {
        case 0: out[idx] = a[idx] + b[idx]; break;
        case 1: out[idx] = a[idx] - b[idx]; break;
        case 2: out[idx] = a[idx] * b[idx]; break;
        case 3: out[idx] = a[idx] / b[idx]; break;
    }
}

__global__ void unary_kernel(const float* in, float* out, size_t n, int op) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = in[idx];
    switch (op) {
        case 0: out[idx] = 1.0f / (1.0f + expf(-x)); break;
        case 1: out[idx] = tanhf(x); break;
        case 2: out[idx] = logf(fmaxf(x, 1e-20f)); break;
    }
}

__global__ void relu_kernel(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = fmaxf(in[idx], 0.0f);
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int TILE = 16;
    __shared__ float sharedA[TILE][TILE];
    __shared__ float sharedB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE - 1) / TILE; ++tile) {
        int a_col = tile * TILE + threadIdx.x;
        int b_row = tile * TILE + threadIdx.y;

        sharedA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        sharedB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void transpose_kernel(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < rows && y < cols) {
        out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void reduce_sum_last_kernel(const float* in, float* out, int outer, int inner) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    int offset = row * inner;

    for (int j = tid; j < inner; j += blockDim.x) {
        sum += in[offset + j];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            shared[tid] += shared[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = shared[0];
    }
}

__global__ void softmax_kernel(const float* in, float* out, int rows, int cols) {
    extern __shared__ float buffer[];
    float* row_max = buffer;
    float* row_sum = buffer + 1;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float max_val = -FLT_MAX;
    for (int j = tid; j < cols; j += stride) {
        max_val = fmaxf(max_val, in[row * cols + j]);
    }

    row_max[0] = max_val;
    __syncthreads();
    max_val = row_max[0];

    float sum = 0.0f;
    for (int j = tid; j < cols; j += stride) {
        float v = expf(in[row * cols + j] - max_val);
        sum += v;
        out[row * cols + j] = v;
    }

    row_sum[0] = sum;
    __syncthreads();
    sum = row_sum[0];

    for (int j = tid; j < cols; j += stride) {
        out[row * cols + j] /= fmaxf(sum, 1e-8f);
    }
}

__global__ void sgd_update_kernel(float* param, const float* grad, float lr, float momentum, float* velocity, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = momentum * velocity[idx] - lr * grad[idx];
    velocity[idx] = v;
    param[idx] += v;
}

__global__ void adam_update_kernel(float* param,
                                   const float* grad,
                                   float lr,
                                   float beta1,
                                   float beta2,
                                   float epsilon,
                                   float* m1,
                                   float* m2,
                                   float bias_correction1,
                                   float bias_correction2,
                                   size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];
    float m = beta1 * m1[idx] + (1.0f - beta1) * g;
    float v = beta2 * m2[idx] + (1.0f - beta2) * g * g;
    m1[idx] = m;
    m2[idx] = v;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
}

void launch_elementwise(const float* a, const float* b, float* out, size_t n, int op, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    elementwise_kernel<<<grid, block, 0, stream>>>(a, b, out, n, op);
    CUDA_CHECK(cudaGetLastError());
}

void launch_unary(const float* in, float* out, size_t n, int op, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    unary_kernel<<<grid, block, 0, stream>>>(in, out, n, op);
    CUDA_CHECK(cudaGetLastError());
}

void add(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    launch_elementwise(a.data(), b.data(), out.data(), a.numel(), 0, stream);
}

void subtract(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    launch_elementwise(a.data(), b.data(), out.data(), a.numel(), 1, stream);
}

void multiply(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    launch_elementwise(a.data(), b.data(), out.data(), a.numel(), 2, stream);
}

void divide(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    launch_elementwise(a.data(), b.data(), out.data(), a.numel(), 3, stream);
}

void relu(const Tensor& in, Tensor& out, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((in.numel() + block - 1) / block);
    relu_kernel<<<grid, block, 0, stream>>>(in.data(), out.data(), in.numel());
    CUDA_CHECK(cudaGetLastError());
}

void sigmoid(const Tensor& in, Tensor& out, cudaStream_t stream) {
    launch_unary(in.data(), out.data(), in.numel(), 0, stream);
}

void tanh(const Tensor& in, Tensor& out, cudaStream_t stream) {
    launch_unary(in.data(), out.data(), in.numel(), 1, stream);
}

void log(const Tensor& in, Tensor& out, cudaStream_t stream) {
    launch_unary(in.data(), out.data(), in.numel(), 2, stream);
}

void matmul(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    int M = a.shape().at(0);
    int K = a.shape().at(1);
    int N = b.shape().at(1);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block, 0, stream>>>(a.data(), b.data(), out.data(), M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void transpose(const Tensor& in, Tensor& out, cudaStream_t stream) {
    int rows = in.shape().at(0);
    int cols = in.shape().at(1);
    dim3 block(32, 8);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);
    transpose_kernel<<<grid, block, 0, stream>>>(in.data(), out.data(), rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void reduce_sum_last_dim(const Tensor& in, Tensor& out, cudaStream_t stream) {
    int outer = in.shape().at(0);
    int inner = in.shape().at(1);
    int block = 256;
    size_t shared_bytes = block * sizeof(float);
    reduce_sum_last_kernel<<<outer, block, shared_bytes, stream>>>(in.data(), out.data(), outer, inner);
    CUDA_CHECK(cudaGetLastError());
}

void softmax(const Tensor& in, Tensor& out, int axis, cudaStream_t stream) {
    int rows = in.shape().at(0);
    int cols = in.shape().at(1);
    int threads = 256;
    softmax_kernel<<<rows, threads, 2 * sizeof(float), stream>>>(in.data(), out.data(), rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void sgd_update(Tensor& param, const Tensor& grad, float lr, float momentum, Tensor& velocity, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((param.numel() + block - 1) / block);
    sgd_update_kernel<<<grid, block, 0, stream>>>(param.data(), grad.data(), lr, momentum, velocity.data(), param.numel());
    CUDA_CHECK(cudaGetLastError());
}

void adam_update(Tensor& param,
                 const Tensor& grad,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 Tensor& m1,
                 Tensor& m2,
                 int64_t timestep,
                 cudaStream_t stream) {
    int block = 256;
    int grid = (int)((param.numel() + block - 1) / block);
    float bias_correction1 = 1.0f - powf(beta1, (float)timestep);
    float bias_correction2 = 1.0f - powf(beta2, (float)timestep);
    adam_update_kernel<<<grid, block, 0, stream>>>(
        param.data(), grad.data(), lr, beta1, beta2, epsilon, m1.data(), m2.data(), bias_correction1, bias_correction2, param.numel());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_ops
} // namespace minidl
