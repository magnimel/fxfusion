#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

__global__ void embedding_kernel(
    const int64_t* __restrict__ idx, 
    const float* __restrict__ w, 
    float* __restrict__ out, 
    int64_t N, int64_t vocab_size, 
    int64_t d_model
) {

    int64_t k = blockIdx.x;
    int64_t id = idx[k];

    float* out_row = out + k * d_model;
    const float* w_row = w + id * d_model;

    for(int i = threadIdx.x; i < d_model; i += blockDim.x) {
        out_row[i] = w_row[i];
    }

}
void embedding (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& idx = reg[input_ids[0]];
    const auto& w  = reg[input_ids[1]];
    auto& out           = reg[output_ids[0]];

    const int64_t* idx_ptr = idx.data_ptr<int64_t>();
    const float* w_ptr = w.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int64_t N = idx.numel();
    int64_t vocab_size = w.size(0);
    int64_t d_model = w.size(1);

    dim3 block(256);
    dim3 grid(N);

    embedding_kernel<<<grid, block>>>(idx_ptr, w_ptr, out_ptr, N, vocab_size, d_model);
    
}

} 