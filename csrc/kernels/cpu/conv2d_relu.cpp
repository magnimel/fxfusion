#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels/cpu/kernels.hpp"

namespace fxfusion::kernels::cpu {

void conv2d_relu (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    TORCH_CHECK(false, "conv2d_relu: not implemented yet");
}

} 