#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

__global__ void relu_kernel(const float* a, float* out, int64_t N) {
    
    for(int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x) 
    {
        out[idx] = fmaxf(0.0f, a[idx]);
    }
} 

void relu(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& a = reg[input_ids[0]];
    auto& out     = reg[output_ids[0]];
    const int64_t N = a.numel();

    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    relu_kernel<<<grid, block>>>(a_ptr, out_ptr, N);
}

}