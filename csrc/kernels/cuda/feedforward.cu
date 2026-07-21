#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void feedforward(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const FeedForwardCache*>(cache_base);

    const auto& x   = reg[input_ids[0]];
    const auto& w1  = reg[input_ids[1]];
    const auto& b1  = reg[input_ids[2]];
    const auto& w2  = reg[input_ids[3]];
    const auto& b2  = reg[input_ids[4]];
    auto& out       = reg[output_ids[0]];

    const float* x_ptr = x.data_ptr<float>();
    const float* w1_ptr = w1.data_ptr<float>();
    const float* b1_ptr = b1.data_ptr<float>();
    const float* w2_ptr = w2.data_ptr<float>();
    const float* b2_ptr = b2.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    float* intermediate_ptr = (cache->data).intermediate;

    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = w1.size(0);
    int64_t P = w2.size(0);

    dim3 block(LINEAR_TILE_SIZE, LINEAR_TILE_SIZE);

    dim3 grid1((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    linear_relu_kernel<<<grid1, block>>>(x_ptr, w1_ptr, b1_ptr, intermediate_ptr, M, N, K);

    dim3 grid2((P + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    linear_kernel<<<grid2, block>>>(intermediate_ptr, w2_ptr, b2_ptr, out_ptr, M, P, N);
    
}

} 