#include "cache.cuh"

namespace fxfusion::kernels::cuda {

MHACache::MHACache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    auto* cuda_ctx = static_cast<GraphCudaContext*>(ctx);
    this->ctx = ctx;
    
    const auto& x        = reg[input_ids[0]];
    const auto& qkv_bias = reg[input_ids[3]];
    const auto& out_bias = reg[input_ids[5]];

    const int64_t num_heads = params.ints[0];
    const int64_t head_dim  = params.ints[1];
    const int64_t d_model   = params.ints[2];
    const int64_t qkv_dim   = params.ints[3];

    const int64_t batch = x.size(0);
    const int64_t seq   = x.size(1);

    qkv_buf        = torch::empty({batch, seq, qkv_dim}, x.options());
    out_ctx_buf    = torch::empty({batch, seq, d_model}, x.options());

    data.qkv       = qkv_buf.data_ptr<float>();
    data.out_ctx   = out_ctx_buf.data_ptr<float>();
    data.batch     = batch;
    data.seq       = seq;

    const void* qkv_b_ptr = qkv_bias.data_ptr<float>();
    const void* out_b_ptr = out_bias.data_ptr<float>();

    int64_t M = batch * seq;

    // --- First Projection: X @ W_qkv^T + b_qkv (NO ReLU) ---
    descSet1.reset(new CublasLtDescriptorSet());
    descSet1->init(cuda_ctx, this->algo1, d_model, M, qkv_dim, qkv_b_ptr, false);

    // --- Second Projection: Attn_Ctx @ W_out^T + b_out (NO ReLU) ---
    descSet2.reset(new CublasLtDescriptorSet());
    descSet2->init(cuda_ctx, this->algo2, d_model, M, d_model, out_b_ptr, false);
}

std::unique_ptr<Cache> build_mha_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<MHACache>(ctx, reg, input_ids, output_ids, params);
}

} // namespace fxfusion::kernels::cuda