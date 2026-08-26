#include "kernels.cuh"
#include <cublas_v2.h>
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
    const auto& b1 = reg[input_ids[2]];
    const auto& w2 = reg[input_ids[3]];
    const auto& b2 = reg[input_ids[4]];
    auto& out      = reg[output_ids[0]];

    const float* x_ptr  = x.data_ptr<float>();
    const float* w1_ptr = w1.data_ptr<float>();
    const float* b1_ptr = b1.data_ptr<float>();
    const float* w2_ptr = w2.data_ptr<float>();
    const float* b2_ptr = b2.data_ptr<float>();
    float* out_ptr      = out.data_ptr<float>();
    float* inter_ptr    = cache->data.intermediate;

    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = w1.size(0);
    int64_t P = w2.size(0);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // -----------------------------------------------------------------
    // 1. First projection:  x @ w1^T + b1  →  intermediate
    //    x: {M, K}  @  w1^T: {K, N}  →  inter: {M, N}
    // -----------------------------------------------------------------
    {
        cublasStatus_t status = cublasSgemm(
            cuda_ctx->cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int32_t>(N), static_cast<int32_t>(M), static_cast<int32_t>(K),
            &alpha,
            w1_ptr, static_cast<int32_t>(K),
            x_ptr,  static_cast<int32_t>(K),
            &beta,
            inter_ptr, static_cast<int32_t>(N)
        );
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "feedforward cublasSgemm [1] failed: ", cublas_get_error_string(status));
    }

    {
        dim3 block(256);
        dim3 grid((M * N + block.x - 1) / block.x);
        add_relu_kernel<<<grid, block>>>(inter_ptr, b1_ptr, inter_ptr, M * N, N);
    }

    // -----------------------------------------------------------------
    // 2. Second projection:  inter @ w2^T + b2  →  out
    //    inter: {M, N}  @  w2^T: {N, P}  →  out: {M, P}
    // -----------------------------------------------------------------
    {
        cublasStatus_t status = cublasSgemm(
            cuda_ctx->cublas_handle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int32_t>(P), static_cast<int32_t>(M), static_cast<int32_t>(N),
            &alpha,
            w2_ptr, static_cast<int32_t>(N),
            inter_ptr, static_cast<int32_t>(N),
            &beta,
            out_ptr, static_cast<int32_t>(P)
        );
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "feedforward cublasSgemm [2] failed: ", cublas_get_error_string(status));
    }

    {
        dim3 block(256);
        dim3 grid((M * P + block.x - 1) / block.x);
        add_kernel<<<grid, block>>>(out_ptr, b2_ptr, out_ptr, M * P, P);
    }
}

} // namespace fxfusion::kernels::cuda