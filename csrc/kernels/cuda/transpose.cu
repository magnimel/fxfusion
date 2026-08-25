#include "kernels.cuh"
#include "internal/transpose_kernel.cuh"

namespace fxfusion::kernels::cuda {

void transpose(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const TransposeCache*>(cache_base);

    const auto& x = reg[input_ids[0]];
    auto& out      = reg[output_ids[0]];
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t N = x.numel();

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    transpose_kernel<<<grid, block>>>(x_ptr, out_ptr, N, cache->data);
}


} // namespace fxfusion::kernels::cuda

