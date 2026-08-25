#pragma once
#include "graph_cuda_context.cuh"
#include "runtime_types.hpp"
#include <torch/torch.h>
#include <cudnn_frontend.h>
#include <cudnn.h>
#include <memory>

namespace fxfusion::kernels::cuda {

namespace fe = cudnn_frontend;

struct LinearCache : public Cache {
    LinearCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
    GraphContext* ctx;
};

struct LinearReluCache : public LinearCache {
    LinearReluCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
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
};

struct Conv2DCacheData {
    int64_t x_uid, w_uid, b_uid, out_uid;
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