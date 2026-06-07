#pragma once
#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "runtime_node.hpp"

namespace fxfusion {

class RuntimeGraph {
public:
    RuntimeGraph(const fxfusion::Graph* graph, const torch::Device& device);
    void execute(TensorRegistry& reg);

private:
    std::vector<RuntimeNode> nodes_;
};

}

