#include "kernels.cuh"
#include "internal/flash_attention_kernel.cuh"

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
    const auto& qkv_bias   = reg[input_ids[3]];
    const auto& out_weight = reg[input_ids[4]];
    const auto& out_bias   = reg[input_ids[5]];
    auto& out              = reg[output_ids[0]];

    const int64_t num_heads     = params.ints[0];
    const int64_t head_dim      = params.ints[1];
    const int64_t d_model       = params.ints[2];
    const int64_t qkv_dim       = params.ints[3];
    const float   scale_divisor = params.floats[0];

    const float* x_ptr     = x.data_ptr<float>();
    const bool*  mask_ptr  = mask.data_ptr<bool>();
    const float* qkv_w_ptr = qkv_weight.data_ptr<float>();
    const float* qkv_b_ptr = qkv_bias.data_ptr<float>();
    const float* out_w_ptr = out_weight.data_ptr<float>();
    const float* out_b_ptr = out_bias.data_ptr<float>();
    float* out_ptr         = out.data_ptr<float>();

    float* qkv     = cache->data.qkv;
    float* out_ctx = cache->data.out_ctx;
    int64_t batch  = cache->data.batch;
    int64_t seq    = cache->data.seq;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    int64_t M = batch * seq;   // rows
    int64_t K = d_model;       // inner dim

    // -----------------------------------------------------------------
    // 1. QKV Projection
    //    x: {M, K}  @  W_qkv^T: {K, qkv_dim}  →  qkv: {M, qkv_dim}
    // -----------------------------------------------------------------
    int64_t N_qkv = qkv_dim;

    {   
        cublasStatus_t status = cublasSgemm(
            cuda_ctx->cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int32_t>(N_qkv), static_cast<int32_t>(M), static_cast<int32_t>(K),
            &alpha,
            qkv_w_ptr, static_cast<int32_t>(K),
            x_ptr,     static_cast<int32_t>(K),
            &beta,
            qkv,       static_cast<int32_t>(N_qkv)
        );
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "mha_flash cublasSgemm [QKV projection] failed: ", cublas_get_error_string(status));
    }

    {
        dim3 block(256);
        dim3 grid((M * N_qkv + block.x - 1) / block.x);
        add_kernel<false><<<grid, block>>>(qkv, qkv_b_ptr, qkv, M * N_qkv, N_qkv);
    }

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
    int64_t N_out = d_model;   

    {
        cublasStatus_t status = cublasSgemm(
            cuda_ctx->cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int32_t>(N_out), static_cast<int32_t>(M), static_cast<int32_t>(K),
            &alpha,
            out_w_ptr, static_cast<int32_t>(K),
            out_ctx,   static_cast<int32_t>(K),
            &beta,
            out_ptr,   static_cast<int32_t>(N_out)
        );
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "mha_flash cublasSgemm [Out projection] failed: ", cublas_get_error_string(status));
    }

    {
        dim3 block(256);
        dim3 grid((M * N_out + block.x - 1) / block.x);
        add_kernel<false><<<grid, block>>>(out_ptr, out_b_ptr, out_ptr, M * N_out, N_out);
    }
}

} // namespace fxfusion::kernels::cuda