#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

__global__ void add_layer_norm_kernel(const float* x, const float* z, const float* w, const float* b, float* out, int64_t M, int64_t N, float eps) {

    extern __shared__ float tmp[];

    const float* x_row = x + blockIdx.x * N;
    const float* z_row = z + blockIdx.x * N;
    float* out_row = out + blockIdx.x * N;    

    int64_t tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int64_t i = tid; i < N; i += blockDim.x) {
        local_sum += x_row[i] + z_row[i];
    }
    tmp[tid] = local_sum;
    __syncthreads();

    for (int64_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) tmp[tid] += tmp[tid + stride];
        __syncthreads();
    }
    __shared__ float mean;
    if (tid == 0) mean = tmp[0] / static_cast<float>(N);
    __syncthreads();

    float local_sumsq = 0.0f;
    for (int64_t i = tid; i < N; i += blockDim.x) {
        float d = (x_row[i] + z_row[i]) - mean;
        local_sumsq += d * d;
    }
    tmp[tid] = local_sumsq;
    __syncthreads();

    __shared__ float inv_std;
    for (int64_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) tmp[tid] += tmp[tid + stride];
        __syncthreads();
    }
    if (tid == 0) inv_std = rsqrtf(tmp[0] / static_cast<float>(N) + eps);
    __syncthreads();

    for (int64_t i = tid; i < N; i += blockDim.x) {
        out_row[i] = (x_row[i] + z_row[i] - mean) * inv_std * w[i] + b[i];
    }
}

void add_layer_norm (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const AddLayerNormCache*>(cache_base);

    const auto& x = reg[input_ids[0]];
    const auto& z = reg[input_ids[1]];
    const auto& w = reg[input_ids[2]];
    const auto& b = reg[input_ids[3]];
    auto& out     = reg[output_ids[0]];

    const float* x_ptr = x.data_ptr<float>();
    const float* z_ptr = z.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const float eps = params.floats[0];
    const int64_t M = (cache->data).M;
    const int64_t N = (cache->data).N;

    dim3 block(256);
    dim3 grid(M);

    size_t shared_bytes = block.x * sizeof(float);
    add_layer_norm_kernel<<<grid, block, shared_bytes>>>(x_ptr, z_ptr, w_ptr, b_ptr, out_ptr, M, N, eps);
}

} 