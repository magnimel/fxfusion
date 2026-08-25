#pragma once
#include <vector>
#include <memory>
#include <torch/torch.h>
#include "graph_generated.h"
#include "memory_manager.hpp"
#include "op_registry.hpp"

namespace fxfusion {

class RuntimeNode {
public:
    RuntimeNode(const fxfusion::Node* node, const OpDef& def, TensorRegistry& reg, const torch::Device& device);
    void execute(TensorRegistry& reg);
    OpCode op_code() const { return op_code_; }
    
private:
    OpCode op_code_;
    KernelFn kernel_;
    TensorIds input_ids_;
    TensorIds output_ids_;
    Params params_;
    std::unique_ptr<Cache> cache_;
};

}