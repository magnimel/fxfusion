#pragma once

#include <string>
#include <vector>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/extension.h>

#include "graph_generated.h"

namespace fxfusion {

class ExecutionEngine : public torch::CustomClassHolder {
public:
    explicit ExecutionEngine(const std::string& graph_path, const std::string& device = "cpu");
    
    std::vector<torch::Tensor> run(const std::vector<torch::Tensor>& inputs);
    
private:
    std::vector<char> buffer_;
    const fxfusion::Graph* graph_ = nullptr;
    torch::Tensor arena_;
    torch::Device device_; 
};
        
} 