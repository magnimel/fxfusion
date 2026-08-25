#pragma once
#include <vector>
#include <cstdint>
#include <torch/torch.h>

namespace fxfusion {

using TensorRegistry = std::vector<torch::Tensor>;
using TensorIds      = std::vector<uint32_t>;

struct Params {
    std::vector<int64_t> ints;
    std::vector<float> floats;
};

struct Cache {
    virtual ~Cache() = default;
};

struct GraphContext {
    virtual ~GraphContext() = default;
    virtual void ensure_workspace() {} 
};

using KernelFn = void (*)(
    TensorRegistry&,
    const TensorIds&,
    const TensorIds&,
    const Params&,
    const Cache*
);

using CacheBuilderFn = std::unique_ptr<Cache> (*)(
    GraphContext* ctx,
    TensorRegistry& reg,
    const TensorIds& input_ids,
    const TensorIds& output_ids,
    const Params& params
);


}