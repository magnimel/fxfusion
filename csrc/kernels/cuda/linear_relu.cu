#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

// =====================================================================
// LINEAR
// =====================================================================
// Input:  x   {M, K}
//         w   {N, K}  (transposed by cublasLtMatmul)
//         b   {N}     (Bias is bound to descSet in cache constructor)
// Output: out {M, N}
//
// Operation: out = ReLU(x @ w^T + b)
// =====================================================================
void linear_relu(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    auto* cache = static_cast<const LinearCache*>(cache_base);
    auto* cuda_ctx = static_cast<GraphCudaContext*>(cache->ctx);
    
    const auto& x = reg[input_ids[0]];
    const auto& w = reg[input_ids[1]];
    const auto& b = reg[input_ids[2]];
    auto& out     = reg[output_ids[0]];
    
    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    
    auto* descSet = cache->descSet.get();
    
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLASLT_CHECK(cublasLtMatmul(
        cuda_ctx->cublasLt_handle(),
        descSet->operationDesc,
        &alpha,
        w_ptr, descSet->Adesc,
        x_ptr, descSet->Bdesc,
        &beta,
        out_ptr, descSet->Cdesc,
        out_ptr, descSet->Cdesc,
        &cache->algo,
        cuda_ctx->workspace_ptr(),
        cuda_ctx->workspace_size(),  
        /* stream */ nullptr));

}

} // namespace fxfusion::kernels::cuda