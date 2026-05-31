#pragma once
#include <vector>
#include <torch/torch.h>
#include "gen/graph_generated.h"

namespace fxfusion {

struct AliasInstruction {
    int32_t id;          
    int32_t source_id;   
    std::vector<int64_t> shape;
};

class MemoryManager {
public:
    MemoryManager(const fxfusion::Graph* graph, torch::Device device);
    void bind_inputs(const std::vector<torch::Tensor>& inputs);
    const std::vector<torch::Tensor>& get_registry() const { return registry_; }

private:
    const fxfusion::Graph* graph_ = nullptr;;   
    torch::Tensor arena_;
    std::vector<torch::Tensor> registry_;
    
    std::vector<int32_t> input_ids_; 
    std::vector<AliasInstruction> aliases_;
};

}