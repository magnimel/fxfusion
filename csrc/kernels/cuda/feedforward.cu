#include "kernels.cuh"
#include "internal/elementwise_kernels.cuh"

namespace fxfusion::kernels::cuda {

// =====================================================================
// FEEDFORWARD
// =====================================================================
// Input:  x    {batch, seq, d_model}
//         w1   {ffn_dim, d_model}   (first projection weight)
//         b1   {ffn_dim}            (first projection bias)
//         w2   {d_model, ffn_dim}   (second projection weight)
//         b2   {d_model}            (second projection bias)
// Output: out  {batch, seq, d_model}
//
// Operation:
//   1. inter = ReLU(x @ w1^T + b1)   → {batch*seq, ffn_dim}
//   2. out   = inter @ w2^T + b2     → {batch*seq, d_model}
// =====================================================================
void feedforward(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const FeedForwardCache*>(cache_base);
    auto* cuda_ctx = static_cast<GraphCudaContext*>(cache->ctx);

    const auto& x  = reg[input_ids[0]];
    const auto& w1 = reg[input_ids[1]];
    const auto& w2 = reg[input_ids[3]];
    auto& out      = reg[output_ids[0]];

    const float* x_ptr  = x.data_ptr<float>();
    const float* w1_ptr = w1.data_ptr<float>();
    const float* w2_ptr = w2.data_ptr<float>();
    float* out_ptr      = out.data_ptr<float>();
    float* inter_ptr    = cache->data.intermediate;

    auto* descSet1 = cache->descSet1.get();
    auto* descSet2 = cache->descSet2.get();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // -----------------------------------------------------------------
    // 1. First projection:  x @ w1^T + b1  →  intermediate
    //    x: {M, K}  @  w1^T: {K, N}  →  inter: {M, N}
    // -----------------------------------------------------------------
    CUBLASLT_CHECK(cublasLtMatmul(
            cuda_ctx->cublasLt_handle(),
            descSet1->operationDesc,
            &alpha,
            w1_ptr, descSet1->Adesc,
            x_ptr, descSet1->Bdesc,
            &beta,
            inter_ptr, descSet1->Cdesc,
            inter_ptr, descSet1->Cdesc, 
            &cache->algo1,
            cuda_ctx->workspace_ptr(),
            cuda_ctx->workspace_size(),
            /* stream */ nullptr
    ));

    // -----------------------------------------------------------------
    // 2. Second projection:  inter @ w2^T + b2  →  out
    //    inter: {M, N}  @  w2^T: {N, P}  →  out: {M, P}
    // -----------------------------------------------------------------
        CUBLASLT_CHECK(cublasLtMatmul(
            cuda_ctx->cublasLt_handle(),
            descSet2->operationDesc,
            &alpha,
            w2_ptr, descSet2->Adesc,
            inter_ptr, descSet2->Bdesc,
            &beta,
            out_ptr, descSet2->Cdesc,
            out_ptr, descSet2->Cdesc, 
            &cache->algo2,
            cuda_ctx->workspace_ptr(),
            cuda_ctx->workspace_size(),
            /* stream */ nullptr
        ));
}

} // namespace fxfusion::kernels::cuda