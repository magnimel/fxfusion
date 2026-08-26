#include "kernels.cuh"
#include "internal/layer_norm_kernel.cuh"

namespace fxfusion::kernels::cuda {

void layer_norm (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const LayerNormCache*>(cache_base);

    const auto& x = reg[input_ids[0]];
    const auto& w = reg[input_ids[1]];
    const auto& b = reg[input_ids[2]];
    auto& out     = reg[output_ids[0]];

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    float eps = params.floats[0];
    int64_t M = (cache->data).M;
    int64_t N = (cache->data).N;

    dim3 block(256);
    dim3 grid(M);

    size_t shared_bytes = block.x * sizeof(float);
    layer_norm_kernel<<<grid, block, shared_bytes>>>(x_ptr, w_ptr, b_ptr, out_ptr, M, N, eps);
}

} // namespace fxfusion::kernels::cuda