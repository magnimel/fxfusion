#include "cache.cuh"
#include <vector>

namespace fxfusion::kernels::cuda {

LayerNormCache::LayerNormCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x = reg[input_ids[0]];
    const int64_t rank = params.ints[0];
    
    std::vector<int64_t> normalized_shape(params.ints.begin() + 1, params.ints.begin() + 1 + rank);
    int64_t N = 1;
    for(int64_t d: normalized_shape) N *= d;
    
    data.N = N;
    data.M = x.numel() / N;
}

AddLayerNormCache::AddLayerNormCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) 
    : LayerNormCache(ctx, reg, input_ids, output_ids, params) {
}

std::unique_ptr<Cache> build_layer_norm_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<LayerNormCache>(ctx, reg, input_ids, output_ids, params);
}

std::unique_ptr<Cache> build_add_layer_norm_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<AddLayerNormCache>(ctx, reg, input_ids, output_ids, params);
}

} // namespace fxfusion::kernels::cuda