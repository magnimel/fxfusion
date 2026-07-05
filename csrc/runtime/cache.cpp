#include "cache.hpp"

namespace fxfusion {

TransposeCache::TransposeCache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
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

FeedForwardCache::FeedForwardCache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x  = reg[input_ids[0]];
    const auto& w1 = reg[input_ids[1]];
    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = w1.size(0);
    intermediate_buf = torch::empty({M, N}, x.options());
    data.intermediate = intermediate_buf.data_ptr<float>();
}

LayerNormCache::LayerNormCache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x = reg[input_ids[0]];
    const int64_t rank = params.ints[0];
    std::vector<int64_t> normalized_shape(params.ints.begin() + 1, params.ints.begin() + 1 + rank);
    int64_t N = 1;
    for(int64_t d: normalized_shape) N *=d;
    data.N = N;
    data.M = x.numel() / N;
}

AddLayerNormCache::AddLayerNormCache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) 
: LayerNormCache(reg, input_ids, output_ids, params) {
    // No additional data members for AddLayerNormCache
}

MHACache::MHACache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x = reg[input_ids[0]];

    const int64_t num_heads = params.ints[0];
    const int64_t head_dim  = params.ints[1];
    const int64_t d_model   = params.ints[2];
    const int64_t qkv_dim   = params.ints[3];

    const int64_t batch = x.size(0);
    const int64_t seq   = x.size(1);

    qkv_buf    = torch::empty({batch, seq, qkv_dim}, x.options());
    scores_buf = torch::empty({batch, num_heads, seq, seq}, x.options());
    ctx_buf    = torch::empty({batch, seq, d_model}, x.options());

    data.qkv    = qkv_buf.data_ptr<float>();
    data.scores = scores_buf.data_ptr<float>();
    data.ctx    = ctx_buf.data_ptr<float>();
    data.batch  = batch;
    data.seq    = seq;
}

    
}