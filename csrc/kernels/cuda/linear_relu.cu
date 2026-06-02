#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void linear_relu (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    TORCH_CHECK(false, "linear_relu: not implemented yet");
}

} 