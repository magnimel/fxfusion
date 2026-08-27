#pragma once
#include "graph_cuda_context.cuh"
#include "runtime_types.hpp"
#include <torch/torch.h>
#include <cudnn_frontend.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <memory>

namespace fxfusion::kernels::cuda {

namespace fe = cudnn_frontend;

struct CublasLtDescriptorSet {
    cublasLtMatmulDesc_t       operationDesc = nullptr;
    cublasLtMatrixLayout_t     Adesc         = nullptr;
    cublasLtMatrixLayout_t     Bdesc         = nullptr;
    cublasLtMatrixLayout_t     Cdesc         = nullptr;
    cublasLtMatmulPreference_t preference    = nullptr;

    void init(GraphCudaContext* ctx, cublasLtMatmulAlgo_t& algo, int64_t K, int64_t M, int64_t N, const void* bias, bool RELU);
};

struct CublasLtDescriptorSetDeleter {
    void operator()(CublasLtDescriptorSet* h) const {
        if (h == nullptr) return;
        if (h->preference)    cublasLtMatmulPreferenceDestroy(h->preference);
        if (h->Cdesc)         cublasLtMatrixLayoutDestroy(h->Cdesc);
        if (h->Bdesc)         cublasLtMatrixLayoutDestroy(h->Bdesc);
        if (h->Adesc)         cublasLtMatrixLayoutDestroy(h->Adesc);
        if (h->operationDesc) cublasLtMatmulDescDestroy(h->operationDesc);
        delete h;
    }
};

using DescriptorSet = std::unique_ptr<CublasLtDescriptorSet, CublasLtDescriptorSetDeleter>;

struct LinearCache : public Cache {
    LinearCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, bool RELU);
    DescriptorSet descSet;
    cublasLtMatmulAlgo_t algo,
    GraphContext* ctx;
};

struct LinearReluCache : public LinearCache {
    LinearReluCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, bool RELU);
};

struct TransposeCacheData {
    static constexpr int kMaxDims = 8;
    int64_t dims;
    int64_t perm[kMaxDims];
    int64_t in_stride[kMaxDims];
    int64_t out_shape[kMaxDims];
    int64_t out_stride[kMaxDims];
};

struct TransposeCache : public Cache {
    TransposeCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
    TransposeCacheData data;
};

struct FeedForwardCacheData {
    float* intermediate;
};

struct FeedForwardCache : public Cache {
    FeedForwardCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
    
    torch::Tensor intermediate_buf; 
    FeedForwardCacheData data;     
    GraphContext* ctx; 
    
    DescriptorSet descSet1;
    DescriptorSet descSet2;
    
    cublasLtMatmulAlgo_t algo1; 
    cublasLtMatmulAlgo_t algo2; 
};

struct LayerNormCacheData {
    int64_t M = 0;       
    int64_t N = 0; 
};

struct LayerNormCache : public Cache {
    LayerNormCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
    LayerNormCacheData data;    
};

struct AddLayerNormCache : public LayerNormCache {
    AddLayerNormCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
};

struct MHACacheData {
    float* qkv;    
    float* out_ctx;    
    int64_t batch;
    int64_t seq;
};

struct MHACache : public Cache {
    MHACache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
    torch::Tensor qkv_buf;
    torch::Tensor out_ctx_buf;
    MHACacheData data; 
    GraphContext* ctx;

    DescriptorSet descSet1;
    DescriptorSet descSet2;
    
    cublasLtMatmulAlgo_t algo1; 
    cublasLtMatmulAlgo_t algo2; 
};

struct Conv2DCacheData {
    std::shared_ptr<fe::graph::Tensor_attributes> X;
    std::shared_ptr<fe::graph::Tensor_attributes> W;
    std::shared_ptr<fe::graph::Tensor_attributes> B;
    std::shared_ptr<fe::graph::Tensor_attributes> Y;
};

struct Conv2DCache : public Cache {
    Conv2DCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, bool RELU);
    std::unique_ptr<fe::graph::Graph> graph;
    Conv2DCacheData data;
    GraphContext* ctx;
};

struct Conv2DReluCache : public Conv2DCache {
    Conv2DReluCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, bool RELU);
};

std::unique_ptr<Cache> build_linear_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
std::unique_ptr<Cache> build_linear_relu_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
std::unique_ptr<Cache> build_transpose_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
std::unique_ptr<Cache> build_feedforward_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
std::unique_ptr<Cache> build_layer_norm_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
std::unique_ptr<Cache> build_add_layer_norm_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
std::unique_ptr<Cache> build_mha_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
std::unique_ptr<Cache> build_conv2d_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
std::unique_ptr<Cache> build_conv2d_relu_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);

} // namespace fxfusion::kernels::cuda