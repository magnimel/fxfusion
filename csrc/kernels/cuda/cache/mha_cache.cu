#include "cache.cuh"

namespace fxfusion::kernels::cuda {

MHACache::MHACache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    this->ctx = ctx;
    
    const auto& x = reg[input_ids[0]];

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
}

std::unique_ptr<Cache> build_mha_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<MHACache>(ctx, reg, input_ids, output_ids, params);
}

} // namespace fxfusion::kernels::cuda