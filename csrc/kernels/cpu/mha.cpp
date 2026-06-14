#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void mha(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x               = reg[input_ids[0]];
    const auto& mask            = reg[input_ids[1]];
    const auto& qkv_weight      = reg[input_ids[2]];
    const auto& qkv_bias        = reg[input_ids[3]];
    const auto& out_weight      = reg[input_ids[4]];
    const auto& out_bias        = reg[input_ids[5]];
    auto& out                   = reg[output_ids[0]];

    const int64_t num_heads     = params.ints[0];
    const int64_t head_dim      = params.ints[1];
    const int64_t d_model       = params.ints[2];
    const int64_t qkv_dim       = params.ints[3];
    const float   scale_divisor = params.floats[0];

    // QKV projection: [batch, seq, 3*d_model]
    auto qkv = torch::linear(x, qkv_weight, qkv_bias);

    const int64_t batch = x.size(0);
    const int64_t seq   = x.size(1);

    // Split and reshape to [batch, heads, seq, head_dim]
    const auto q = qkv.slice(2, 0,           d_model)
                   .view({batch, seq, num_heads, head_dim})
                   .transpose(1, 2);
    const auto k = qkv.slice(2, d_model,     2 * d_model)
                   .view({batch, seq, num_heads, head_dim})
                   .transpose(1, 2);
    const auto v = qkv.slice(2, 2 * d_model, qkv_dim)
                   .view({batch, seq, num_heads, head_dim})
                   .transpose(1, 2);

    // Attention scores
    auto scores = torch::matmul(q, k.transpose(-2, -1)) / scale_divisor;
    
    scores = scores.masked_fill(mask == 0, -std::numeric_limits<float>::infinity());

    auto attn = torch::softmax(scores, -1);

    // Weighted sum + reshape
    auto ctx = torch::matmul(attn, v)
                   .transpose(1, 2)
                   .contiguous()
                   .view({batch, seq, d_model});

    out.copy_(torch::linear(ctx, out_weight, out_bias));
}

} 