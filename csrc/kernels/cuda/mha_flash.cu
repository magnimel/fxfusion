#include "kernels.cuh"
#include "internal/flash_attention_kernel.cuh"
#include "internal/elementwise_kernels.cuh"

namespace fxfusion::kernels::cuda {

// =====================================================================
// MHA FLASH (HOST)
// =====================================================================
// Input:  x          {batch, seq, d_model}
//         mask       {batch, 1, seq, seq} bool
//         qkv_weight {qkv_dim, d_model}             (QKV projection weight)
//         qkv_bias   {qkv_dim}                      (QKV projection bias)
//         out_weight {d_model, d_model}             (output projection weight)
//         out_bias   {d_model}                      (output projection bias)
// Output: out        {batch, seq, d_model}
//
// Operation:
//   1. QKV projection:  x @ W_qkv^T + b_qkv          → {batch, seq, qkv_dim}
//   2. Flash Attention: softmax(QK^T/√d_k)V          → {batch, seq, d_model}
//   3. Out projection:  attn_out @ W_out^T + b_out   → {batch, seq, d_model}
// =====================================================================
void mha_flash(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
const auto* cache = static_cast<const MHACache*>(cache_base);
    auto* cuda_ctx = static_cast<GraphCudaContext*>(cache->ctx);

    const auto& x          = reg[input_ids[0]];
    const auto& mask       = reg[input_ids[1]];
    const auto& qkv_weight = reg[input_ids[2]];
    const auto& out_weight = reg[input_ids[4]];
    auto& out              = reg[output_ids[0]];

    int64_t num_heads      = params.ints[0];
    int64_t head_dim       = params.ints[1];
    int64_t d_model        = params.ints[2];
    int64_t qkv_dim        = params.ints[3];
    float   scale_divisor  = params.floats[0];

    const float* x_ptr     = x.data_ptr<float>();
    const bool*  mask_ptr  = mask.data_ptr<bool>();
    const float* qkv_w_ptr = qkv_weight.data_ptr<float>();
    const float* out_w_ptr = out_weight.data_ptr<float>();
    float* out_ptr         = out.data_ptr<float>();

    float* qkv     = cache->data.qkv;
    float* out_ctx = cache->data.out_ctx;
    int64_t batch  = cache->data.batch;
    int64_t seq    = cache->data.seq;

    auto* descSet1 = cache->descSet1.get();
    auto* descSet2 = cache->descSet2.get();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // -----------------------------------------------------------------
    // 1. QKV Projection
    //    x: {M, K}  @  W_qkv^T: {K, qkv_dim}  →  qkv: {M, qkv_dim}
    // -----------------------------------------------------------------
    CUBLASLT_CHECK(cublasLtMatmul(
        cuda_ctx->cublasLt_handle(),
        descSet1->operationDesc,
        &alpha,
        qkv_w_ptr, descSet1->Adesc,
        x_ptr, descSet1->Bdesc,
        &beta,
        qkv, descSet1->Cdesc,
        qkv, descSet1->Cdesc,
        &cache->algo1,
        cuda_ctx->workspace_ptr(),
        cuda_ctx->workspace_size(),
        /* stream */ nullptr
    ));

    // -----------------------------------------------------------------
    // 2. Flash Attention
    //    qkv: {batch, seq, qkv_dim}  →  out_ctx: {batch, seq, d_model}
    // -----------------------------------------------------------------
    {
        dim3 block(FLASH_BLOCK_SIZE, FLASH_BLOCK_SIZE);
        dim3 grid((head_dim + block.x - 1) / block.x, (seq + block.y - 1) / block.y, batch * num_heads);
        flash_attention_kernel<<<grid, block>>>(qkv, mask_ptr, out_ctx,
            batch, seq, num_heads, head_dim, d_model, qkv_dim, scale_divisor);
    }

    // -----------------------------------------------------------------
    // 3. Output Projection
    //    out_ctx: {M, K}  @  W_out^T: {K, d_model}  →  out: {M, d_model}
    // -----------------------------------------------------------------
    CUBLASLT_CHECK(cublasLtMatmul(
        cuda_ctx->cublasLt_handle(),
        descSet2->operationDesc,
        &alpha,
        out_w_ptr, descSet2->Adesc,
        out_ctx, descSet2->Bdesc,
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