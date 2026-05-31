#include <Python.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <fstream>
#include <iostream>
#include "runtime/execution_engine.hpp"
#include "runtime/op_registry.hpp"

namespace fxfusion {

ExecutionEngine::ExecutionEngine(const std::string& graph_path, const std::string& device) 
    : device_(torch::Device(device)) {
    
    std::ifstream file(graph_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("ExecutionEngine failed to open: " + graph_path);
    }

    const auto size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    buffer_.resize(size);
    file.read(buffer_.data(), static_cast<std::streamsize>(size));

    graph_ = fxfusion::GetGraph(buffer_.data());
    memory_manager_ = std::make_unique<MemoryManager>(graph_, device_);
    op_registry_ = std::make_unique<OpRegistry>(device_);

}

std::vector<torch::Tensor> ExecutionEngine::run(const std::vector<torch::Tensor>& inputs) {
    TORCH_CHECK(inputs.size() > 0, "Inputs vector is empty");
    TORCH_CHECK(graph_ != nullptr, "Execution graph is not loaded");
    TORCH_CHECK(inputs[0].device() == device_, "Device mismatched between input and Engine");
    
    memory_manager_->bind_inputs(inputs);
    const auto& registry = memory_manager_->get_registry();
    
    for (const auto* node : *graph_->nodes()) {
        const auto op_code = node->opcode();
        if (op_code == fxfusion::OpCode_Placeholder) {
            continue;
        }

        if (op_code == fxfusion::OpCode_View) {
            
            continue;
        }

        const auto& kernel = op_registry_->get(node->opcode());
        kernel(registry, node);
    }
    
    std::vector<torch::Tensor> outputs;
    for (const auto* _tensor : *graph_->tensors()) {
        if (_tensor->kind() == fxfusion::TensorKind_Output) {
            outputs.push_back(registry[_tensor->id()]);
        }
    }

    return outputs;
}

}