#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void max_pool2d (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    TORCH_CHECK(false, "max_pool2d: not implemented yet");
}

} 