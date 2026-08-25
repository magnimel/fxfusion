#pragma once
#include "kernels.cuh"

// =====================================================================
// RETIRED — reference/historical implementation only.
//
// Host-side reference transpose kept for verification against the
// active transpose_kernel. NOT a CUDA kernel; NOT safe to call against
// device-resident tensors.
//
// Wrapped in #if 0 to prevent compilation while maintaining IDE syntax
// highlighting.
// =====================================================================

#if 0

namespace fxfusion::kernels::cuda {

void transpose_ref(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const TransposeCache*>(cache_base);
    const auto& d = cache->data;

    const auto& x = reg[input_ids[0]];
    auto& out      = reg[output_ids[0]];
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t N = x.numel();

    for (int64_t flat_out = 0; flat_out < N; flat_out++) {
        int64_t flat_in = 0;
        for (int64_t dim = 0; dim < d.dims; dim++) {
            int64_t k = d.perm[dim];
            int64_t idx = (flat_out / d.out_stride[k]) % d.out_shape[k];
            flat_in += idx * d.in_stride[dim];
        }
        out_ptr[flat_out] = x_ptr[flat_in];
    }
}

} // namespace fxfusion::kernels::cuda

#endif