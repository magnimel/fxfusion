#pragma once
#include <string>
#include <vector>
#include <memory>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/extension.h>
#include "memory_manager.hpp"
#include "op_registry.hpp"
#include "runtime_graph.hpp"
#include "graph_generated.h"


namespace fxfusion {

class ExecutionEngine : public torch::CustomClassHolder {
public:
    explicit ExecutionEngine(const std::string& graph_path, const std::string& device = "cpu");
    std::vector<torch::Tensor> run(const std::vector<torch::Tensor>& inputs);

private:
    std::vector<char> buffer_;
    const torch::Device device_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<RuntimeGraph> runtime_graph_;
};

}