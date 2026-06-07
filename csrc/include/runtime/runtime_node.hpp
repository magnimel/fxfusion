#pragma once
#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "memory_manager.hpp"
#include "op_registry.hpp"

namespace fxfusion {

class RuntimeNode {
public:
    RuntimeNode(const fxfusion::Node* node, KernelFn kernel);
    void execute(TensorRegistry& reg);
    OpCode op_code() const { return op_code_; }
    
private:
    OpCode op_code_;
    KernelFn kernel_;
    TensorIds input_ids_;
    TensorIds output_ids_;
    Params params_;
};

}



