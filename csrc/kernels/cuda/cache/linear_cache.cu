#include "cache.cuh"

namespace fxfusion::kernels::cuda {

void CublasLtDescriptorSet::init(GraphCudaContext* ctx, cublasLtMatmulAlgo_t& algo, int64_t K, int64_t M, int64_t N, const void* bias, bool RELU) {
    size_t max_workspace_size_bytes = 32 * 1024 * 1024;
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtEpilogue_t epi = RELU ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS;

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size_bytes, sizeof(max_workspace_size_bytes)));

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, static_cast<uint64_t>(K), static_cast<uint64_t>(N), K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, static_cast<uint64_t>(K), static_cast<uint64_t>(M), K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, static_cast<uint64_t>(N), static_cast<uint64_t>(M), N));

    int32_t returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ctx->cublasLt_handle(),
        operationDesc,
        Adesc,
        Bdesc,
        Cdesc,
        Cdesc, 
        preference,
        1,
        &heuristicResult,
        &returnedResults
    ));

    TORCH_CHECK(returnedResults > 0, "no cublasLt heuristic algo");
    TORCH_CHECK(heuristicResult.state == CUBLAS_STATUS_SUCCESS, "best algo not usable");

    algo = heuristicResult.algo;
    ctx->update_workspace_max_size(heuristicResult.workspaceSize);
}

LinearCache::LinearCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, bool RELU) {
    auto* cuda_ctx = static_cast<GraphCudaContext*>(ctx);
    this->ctx = ctx;

    const auto& x = reg[input_ids[0]];
    const auto& w = reg[input_ids[1]];
    const auto& b = reg[input_ids[2]];
    auto& out     = reg[output_ids[0]];

    const void* b_ptr = b.data_ptr<float>();

    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = w.size(0);
    
    descSet.reset(new CublasLtDescriptorSet());
    descSet->init(cuda_ctx, this->algo, K, M, N, b_ptr, RELU);
}

LinearReluCache::LinearReluCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, bool RELU)
    : LinearCache(ctx, reg, input_ids, output_ids, params, RELU) {
}

std::unique_ptr<Cache> build_linear_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<LinearCache>(ctx, reg, input_ids, output_ids, params, false);
}

std::unique_ptr<Cache> build_linear_relu_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<LinearReluCache>(ctx, reg, input_ids, output_ids, params, true);
}

} // namespace fxfusion::kernels::cuda