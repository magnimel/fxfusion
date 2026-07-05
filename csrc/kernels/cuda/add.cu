#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

__global__ void add_kernel(const float* a, const float* b, float* out, int64_t N) {
    
    for(int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x) 
    {
        out[idx] = a[idx] + b[idx];
    }
} 

void add (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& a = reg[input_ids[0]];
    const auto& b = reg[input_ids[1]];
    auto& out     = reg[output_ids[0]];
    const int64_t N = a.numel();

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    add_kernel<<<grid, block>>>(a_ptr, b_ptr, out_ptr, N);
}

}