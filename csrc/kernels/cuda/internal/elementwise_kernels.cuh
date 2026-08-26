#pragma once

namespace fxfusion::kernels::cuda {

__global__ void relu_kernel(const float* a, float* out, int64_t N) {
    
    for(int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x) 
    {
        out[idx] = fmaxf(0.0f, a[idx]);
    }
} 

template<bool RELU>
__global__ void add_kernel(const float* a, const float* b, float* out, int64_t M, int64_t N) {
    
    for(int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < M;
        idx += blockDim.x * gridDim.x) 
    {
        float res = a[idx] + b[idx % N];
        out[idx] = RELU ? fmaxf(0.0f, res) : res;
    }
}

__global__ void mul_tensor_scalar_kernel(const float* x, float* out, float scalar, int64_t N) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        out[idx] = x[idx] * scalar;
    }
}

__global__ void mul_tensor_tensor_kernel(const float* x, const float* y, float* out, int64_t N) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        out[idx] = x[idx] * y[idx];
    }
}

} // namespace fxfusion::kernels::cuda