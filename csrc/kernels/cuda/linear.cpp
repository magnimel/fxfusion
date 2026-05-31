#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels/cuda/kernels.hpp"

namespace fxfusion::kernels::cuda {

void linear (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node) {
    TORCH_CHECK(false, "linear: not implemented yet");
}

} 