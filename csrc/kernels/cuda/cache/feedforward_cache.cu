#include "cache.cuh"

namespace fxfusion::kernels::cuda {

FeedForwardCache::FeedForwardCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    this->ctx = ctx;
    
    const auto& x  = reg[input_ids[0]];
    const auto& w1 = reg[input_ids[1]];
    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = w1.size(0);
    
    intermediate_buf = torch::empty({M, N}, x.options());
    data.intermediate = intermediate_buf.data_ptr<float>();
}

std::unique_ptr<Cache> build_feedforward_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<FeedForwardCache>(ctx, reg, input_ids, output_ids, params);
}

} // namespace fxfusion::kernels::cuda