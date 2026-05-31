#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels/cuda/kernels.hpp"

namespace fxfusion::kernels::cuda {

void adaptive_avg_pool2d (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    TORCH_CHECK(false, "adaptive_avg_pool2d: not implemented yet");
}

} 