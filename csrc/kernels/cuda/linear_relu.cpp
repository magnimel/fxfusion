#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels/cuda/kernels.hpp"

namespace fxfusion::kernels::cuda {

void linear_relu (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    const auto& x = registry[node->inputs()->Get(0)];
    const auto& w = registry[node->inputs()->Get(1)];
    const auto& b = registry[node->outputs()->Get(2)];
    const auto& out = registry[node->outputs()->Get(0)];
    out.copy_(torch::linear(x, w, b));
}

} 