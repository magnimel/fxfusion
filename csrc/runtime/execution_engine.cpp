#include <Python.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <torch/extension.h>
#include "runtime/execution_engine.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

namespace fxfusion {

ExecutionEngine::ExecutionEngine(const std::string& graph_path, const std::string& device) : device_(torch::Device(device)) {
        std::ifstream file(graph_path, std::ios::binary | std::ios::ate);

        if (!file.is_open()) {
            throw std::runtime_error("ExecutionEngine failed to open: " + graph_path);
        }

        const auto size = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);

        buffer_.resize(size);
        file.read(buffer_.data(), static_cast<std::streamsize>(size));

        graph_ = fxfusion::GetGraph(buffer_.data());

        if (graph_->arena_size() > 0) {
            arena_ = torch::empty(
                {static_cast<int64_t>(graph_->arena_size())},
                torch::TensorOptions().device(device_).dtype(torch::kUInt8)
            );
        }
    }

std::vector<torch::Tensor> ExecutionEngine::run(const std::vector<torch::Tensor>& inputs) {
    std::vector<torch::Tensor> outputs;

    TORCH_CHECK(inputs.size() > 0, "Inputs vector is empty");
    TORCH_CHECK(graph_ != nullptr, "Execution graph is not loaded");
    TORCH_CHECK(inputs[0].device() == device_, "Device mismatched between input and Engine");

    torch::Tensor a = torch::ones({2, 2});
    outputs.push_back(a);
    return outputs;
}



}



