#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void add_relu (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    TORCH_CHECK(false, "add_relu: not implemented yet");
}

} 