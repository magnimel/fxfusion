#pragma once
#include "tensor_registry.hpp"

namespace fxfusion {

struct TransposeCacheData {
    static constexpr int kMaxDims = 8;
    int64_t dims;
    int64_t perm[kMaxDims];
    int64_t in_stride[kMaxDims];
    int64_t out_shape[kMaxDims];
    int64_t out_stride[kMaxDims];
};

struct TransposeCache : public Cache {
    TransposeCache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
    TransposeCacheData data;
};

struct FeedForwardCacheData {
    float* intermediate;
};

struct FeedForwardCache : public Cache {
    FeedForwardCache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
    torch::Tensor intermediate_buf; 
    FeedForwardCacheData data;      
};


struct LayerNormCacheData {
    int64_t M = 0;       
    int64_t N = 0; 
};

struct LayerNormCache : public Cache {
    LayerNormCache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
    LayerNormCacheData data;    
};

struct AddLayerNormCache : public LayerNormCache {
    AddLayerNormCache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
};

struct MHACacheData {
    float* qkv;    
    float* scores; 
    float* ctx;    
    int64_t batch;
    int64_t seq;
};

struct MHACache : public Cache {
    MHACache(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params);
    torch::Tensor qkv_buf;
    torch::Tensor scores_buf;
    torch::Tensor ctx_buf;
    MHACacheData data; 
};


}