#include <Python.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <fstream>
#include <iostream>
#include "execution_engine.hpp"
#include "op_registry.hpp"

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

    const auto* graph = fxfusion::GetGraph(buffer_.data());
    memory_manager_ = std::make_unique<MemoryManager>(graph, device_);
    runtime_graph_ = std::make_unique<RuntimeGraph>(graph, memory_manager_->get_registry(), device_);

}

const std::vector<torch::Tensor>& ExecutionEngine::run(const std::vector<torch::Tensor>& inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        TORCH_CHECK(
            inputs[i].device() == device_, 
            "input engine device mismatch, index: ", i
        );
    }

    memory_manager_->bind_inputs(inputs);
    auto& registry = memory_manager_->get_registry();  
    runtime_graph_->execute(registry);
    return memory_manager_->get_outputs();
}

}