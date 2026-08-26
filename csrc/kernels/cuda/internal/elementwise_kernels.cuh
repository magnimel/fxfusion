#pragma once

namespace fxfusion::kernels::cuda {

__global__ void relu_kernel(const float* a, float* out, int64_t N);

__global__ void add_kernel(const float* a, const float* b, float* out, int64_t M, int64_t N);

__global__ void add_relu_kernel(const float* a, const float* b, float* out, int64_t M, int64_t N);

__global__ void mul_tensor_scalar_kernel(const float* x, float* out, float scalar, int64_t N);

__global__ void mul_tensor_tensor_kernel(const float* x, const float* y, float* out, int64_t N);

} // namespace fxfusion::kernels::cuda