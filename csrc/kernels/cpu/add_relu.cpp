#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels/cpu/kernels.hpp"

namespace fxfusion::kernels::cpu {

void add_relu (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    const auto& a = registry[node->inputs()->Get(0)];
    const auto& b = registry[node->inputs()->Get(1)];
    const auto& out = registry[node->outputs()->Get(0)];
    out.copy_(torch::relu(torch::add(a, b)));
}

} 