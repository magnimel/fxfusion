#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels/cuda/kernels.hpp"

namespace fxfusion::kernels::cuda {

void view (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    TORCH_CHECK(false, "view: not implemented yet");
}

} 