#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

__global__ void add_relu_kernel(const float* a, const float* b, float* out, int64_t M, int64_t N) {
    
    for(int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < M;
        idx += blockDim.x * gridDim.x) 
    {
        out[idx] = fmaxf(0.0f, a[idx] + b[idx % N]);
    }
} 

void add_relu (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& a = reg[input_ids[0]];
    const auto& b = reg[input_ids[1]];
    auto& out     = reg[output_ids[0]];
    const int64_t M = a.numel();
    const int64_t N = b.numel();

    TORCH_CHECK(N > 0 && M % N == 0,
                "add_relu[cuda]: broadcast requires b's numel to evenly divide a's numel, got a.numel()=", M,
                " b.numel()=", N);

    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    dim3 block(256);
    dim3 grid((M + block.x - 1) / block.x);
    add_relu_kernel<<<grid, block>>>(a_ptr, b_ptr, out_ptr, M, N);
}

}