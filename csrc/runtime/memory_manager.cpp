#include "runtime/memory_manager.hpp"

namespace fxfusion {

inline torch::ScalarType get_dtype(const fxfusion::Tensor* tensor) {
    switch (tensor->dtype()) {
        case fxfusion::DType_Float32: return torch::kFloat32;
        case fxfusion::DType_Float16: return torch::kFloat16;
        case fxfusion::DType_Int32:   return torch::kInt32;
        case fxfusion::DType_Int64:   return torch::kInt64;
        default: throw std::runtime_error("Unsupported DType in FlatBuffer");
    }
}

inline std::vector<int64_t> get_shape(const fxfusion::Tensor* tensor) {
    std::vector<int64_t> shape;
    for(auto dim : *tensor->shape()) shape.push_back(dim);
    return shape;
}

MemoryManager::MemoryManager(const fxfusion::Graph* graph, torch::Device device) : graph_(graph) {
    TORCH_CHECK(graph_ != nullptr, "Execution graph is not loaded");

    if (graph_->arena_size() > 0) {
        arena_ = torch::empty(
            {static_cast<int64_t>(graph_->arena_size())},
            torch::TensorOptions().device(device).dtype(torch::kUInt8)
        );
    }

    registry_.resize(graph_->tensors()->size());

    for(const auto* tensor : *graph_->tensors()) {
        auto id = tensor->id();
        auto kind = tensor->kind();
        auto dtype = get_dtype(tensor);
        auto shape = get_shape(tensor);
        
        if (kind == fxfusion::TensorKind_Constant) {
            TORCH_CHECK(tensor->data() != nullptr, "Constant tensor missing data");
            
            auto _cpu_tensor = torch::from_blob(
                (void*) tensor->data()->data(), shape, 
                torch::TensorOptions().device(torch::kCPU).dtype(dtype)
            );
            registry_[id] = _cpu_tensor.to(device);

        }  else if (kind == fxfusion::TensorKind_Activation || kind == fxfusion::TensorKind_Output) {
            TORCH_CHECK(arena_.defined(), "Arena tensor is not allocated")

            size_t offset = static_cast<size_t>(tensor->offset());
            uint8_t* raw_ptr = arena_.data_ptr<uint8_t>();
            
            registry_[id] = torch::from_blob(
                raw_ptr + offset, shape, 
                torch::TensorOptions().device(device).dtype(dtype)
            );

        } else if (kind == fxfusion::TensorKind_Input) {
            input_ids_.push_back(id);

        } else if (kind == fxfusion::TensorKind_Alias) {
            aliases_.push_back(AliasInstruction{ id,
                tensor->alias_of(), get_shape(tensor)
            });
        }
    }
}

void MemoryManager::bind_inputs(const std::vector<torch::Tensor>& inputs) {
    TORCH_CHECK(inputs.size() == input_ids_.size(), "Input count mismatch");

    for (size_t i = 0; i < inputs.size(); ++i) {
        registry_[input_ids_[i]] = inputs[i];
    }

    for (const auto& alias : aliases_) {
        registry_[alias.id] = registry_[alias.source_id].view(alias.shape);
    }
}

}