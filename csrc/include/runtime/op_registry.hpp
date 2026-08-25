#pragma once
#include <functional>
#include <vector>
#include <torch/torch.h>
#include "runtime_types.hpp"
#include "graph_generated.h"

namespace fxfusion {

struct OpDef {
    KernelFn        kernel        = nullptr;
    CacheBuilderFn  cache_builder = nullptr;  // optional
};

class OpRegistry {
public:
    explicit OpRegistry(const torch::Device& device);

    void register_op(OpCode op_code, KernelFn kernel, CacheBuilderFn builder = nullptr);
    const OpDef& get(OpCode op_code) const;

private:
    std::vector<OpDef> registry_;
};

}