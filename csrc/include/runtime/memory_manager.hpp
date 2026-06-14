#pragma once
#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "tensor_registry.hpp"

namespace fxfusion {

class MemoryManager {
public:
    MemoryManager(const fxfusion::Graph* graph, const torch::Device& device);
    void bind_inputs(const std::vector<torch::Tensor>& inputs);
    TensorRegistry& get_registry() { return registry_; }
    const std::vector<torch::Tensor>& get_outputs() const;

private:
    torch::Tensor arena_;
    TensorRegistry registry_;
    TensorIds input_ids_; 
    TensorIds output_ids_; 
    std::vector<torch::Tensor> outputs_;
};

}