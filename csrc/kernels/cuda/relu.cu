#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void relu (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    TORCH_CHECK(false, "relu: not implemented yet");
}

} 