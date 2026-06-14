#pragma once
#include <functional>
#include <vector>
#include <torch/torch.h>
#include "tensor_registry.hpp"
#include "graph_generated.h"

namespace fxfusion {

struct KernelSet {
    KernelFn conv2d;
    KernelFn conv2d_relu;
    KernelFn linear;
    KernelFn linear_relu;
    KernelFn add;
    KernelFn add_relu;
    KernelFn relu;
    KernelFn mul;
    KernelFn max_pool2d;
    KernelFn avg_pool2d;
    KernelFn adaptive_avg_pool2d;
    KernelFn transpose;
    KernelFn size;
    KernelFn narrow;
    KernelFn embedding;
    KernelFn layer_norm;
    KernelFn add_layer_norm;
    KernelFn mha;
    KernelFn feedforward;
};

class OpRegistry {
public:
    explicit OpRegistry(const torch::Device& device);

    void register_op(fxfusion::OpCode op_code, KernelFn fn);
    KernelFn get(fxfusion::OpCode op_code) const;

private:
    std::vector<KernelFn> registry_;
};

}