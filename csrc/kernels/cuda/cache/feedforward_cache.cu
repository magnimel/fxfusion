#include "cache.cuh"

namespace fxfusion::kernels::cuda {

FeedForwardCache::FeedForwardCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    auto* cuda_ctx = static_cast<GraphCudaContext*>(ctx);
    this->ctx = ctx;

    const auto& x  = reg[input_ids[0]];
    const auto& w1 = reg[input_ids[1]];
    const auto& b1 = reg[input_ids[2]];
    const auto& w2 = reg[input_ids[3]];
    const auto& b2 = reg[input_ids[4]];

    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = w1.size(0);
    int64_t P = w2.size(0);

    intermediate_buf = torch::empty({M, N}, x.options());
    data.intermediate = intermediate_buf.data_ptr<float>();

    const void* b1_ptr = b1.data_ptr<float>();
    const void* b2_ptr = b2.data_ptr<float>();

    // --- First Projection: X @ W1^T + B1 (WITH ReLU) ---
    descSet1.reset(new CublasLtDescriptorSet());
    descSet1->init(cuda_ctx, this->algo1, K, M, N, b1_ptr, true);

    // --- Second Projection: Intermediate @ W2^T + B2 (NO ReLU) ---
    descSet2.reset(new CublasLtDescriptorSet());
    descSet2->init(cuda_ctx, this->algo2, N, M, P, b2_ptr, false);
}

std::unique_ptr<Cache> build_feedforward_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<FeedForwardCache>(ctx, reg, input_ids, output_ids, params);
}

} // namespace fxfusion::kernels::cuda