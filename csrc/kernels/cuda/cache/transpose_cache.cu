#include "cache.cuh"

namespace fxfusion::kernels::cuda {

TransposeCache::TransposeCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x   = reg[input_ids[0]];
    const auto& out = reg[output_ids[0]];

    int64_t dim0 = params.ints[0];
    int64_t dim1 = params.ints[1];
    data.dims = x.dim();

    TORCH_CHECK(data.dims <= TransposeCacheData::kMaxDims, "transpose dims exceed cache capacity");

    for (int64_t d = 0; d < data.dims; d++) data.perm[d] = d;
    std::swap(data.perm[dim0], data.perm[dim1]);

    auto in_strides  = x.strides();
    auto out_shape   = out.sizes();
    auto out_strides = out.strides();

    for (int64_t d = 0; d < data.dims; d++) {
        data.in_stride[d]  = in_strides[d];
        data.out_shape[d]  = out_shape[d];
        data.out_stride[d] = out_strides[d];
    }
}

std::unique_ptr<Cache> build_transpose_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<TransposeCache>(ctx, reg, input_ids, output_ids, params);
}

} // namespace fxfusion::kernels::cuda