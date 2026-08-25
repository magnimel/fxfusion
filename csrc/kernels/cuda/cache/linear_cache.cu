#include "cache.cuh"

namespace fxfusion::kernels::cuda {

LinearCache::LinearCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    this->ctx = ctx;
}

LinearReluCache::LinearReluCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params)
    : LinearCache(ctx, reg, input_ids, output_ids, params) {
}

std::unique_ptr<Cache> build_linear_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<LinearCache>(ctx, reg, input_ids, output_ids, params);
}

std::unique_ptr<Cache> build_linear_relu_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<LinearReluCache>(ctx, reg, input_ids, output_ids, params);
}

} // namespace fxfusion::kernels::cuda