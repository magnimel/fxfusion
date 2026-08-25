#include "kernels.cuh"
#include "internal/add_layer_norm_kernel.cuh"

namespace fxfusion::kernels::cuda {

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

    TORCH_CHECK(x.numel() == M * N && z.numel() == M * N,
                "add_layer_norm[cuda]: x/z shape mismatch, expected numel=", M * N,
                " got x.numel()=", x.numel(), " z.numel()=", z.numel());

    dim3 block(256);
    dim3 grid(M);

    size_t shared_bytes = block.x * sizeof(float);
    add_layer_norm_kernel<<<grid, block, shared_bytes>>>(x_ptr, z_ptr, w_ptr, b_ptr, out_ptr, M, N, eps);
}

} // namespace fxfusion::kernels::cuda