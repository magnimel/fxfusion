#pragma once

#include <functional>
#include <vector>

#include <torch/torch.h>

#include "graph_generated.h"

namespace fxfusion {

using OpFn = std::function<void(
    const std::vector<torch::Tensor>& registry,
    const fxfusion::Node* node
)>;

struct KernelSet {
    OpFn conv2d;
    OpFn conv2d_relu;
    OpFn linear;
    OpFn linear_relu;
    OpFn add;
    OpFn add_relu;
    OpFn relu;
    OpFn max_pool2d;
    OpFn avg_pool2d;
    OpFn view;
    OpFn adaptive_avg_pool2d;
};

class OpRegistry {
public:
    explicit OpRegistry(const torch::Device& device);

    void register_op(fxfusion::OpCode op_code, OpFn fn);

    const OpFn& get(fxfusion::OpCode op_code) const;

private:
    std::vector<OpFn> registry_;
};

} 