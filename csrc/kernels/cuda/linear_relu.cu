#include "kernels.cuh"
#include <cublas_v2.h>
#include "internal/elementwise_kernels.cuh"

namespace fxfusion::kernels::cuda {

// =====================================================================
// LINEAR
// =====================================================================
// Input:  x   {M, K}
//         w   {N, K}  (transposed by cublasSgemm)
//         b   {N}
// Output: out {M, N}
//
// Operation: out = ReLU(x @ w^T + b)
// =====================================================================
void linear_relu(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    auto* cache = static_cast<const LinearReluCache*>(cache_base);
    auto* cuda_ctx = static_cast<GraphCudaContext*>(cache->ctx);

    const auto& x = reg[input_ids[0]];
    const auto& w = reg[input_ids[1]];
    const auto& b = reg[input_ids[2]];
    auto& out     = reg[output_ids[0]];

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = w.size(0);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasStatus_t status = cublasSgemm(
        cuda_ctx->cublas_handle(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int32_t>(N), static_cast<int32_t>(M), static_cast<int32_t>(K),
        &alpha,
        w_ptr, static_cast<int32_t>(K),
        x_ptr, static_cast<int32_t>(K),
        &beta,
        out_ptr, static_cast<int32_t>(N)
    );
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "linear_relu cublasSgemm failed: ", cublas_get_error_string(status));

    dim3 block(256);
    dim3 grid((M * N + block.x - 1) / block.x);
    add_kernel<true><<<grid, block>>>(out_ptr, b_ptr, out_ptr, M * N, N);
    
}

} // namespace fxfusion::kernels::cuda