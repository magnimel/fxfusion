#include "memory_manager.hpp"
#include "op_registry.hpp"

namespace fxfusion {

inline torch::ScalarType get_dtype(const fxfusion::Tensor* tensor) {
    switch (tensor->dtype()) {
        case fxfusion::DType_Float32: return torch::kFloat32;
        case fxfusion::DType_Float16: return torch::kFloat16;
        case fxfusion::DType_Int32:   return torch::kInt32;
        case fxfusion::DType_Int64:   return torch::kInt64;
        case fxfusion::DType_Bool:    return torch::kBool;
        default:
            throw std::runtime_error("Unsupported DType in FlatBuffer");
    }
}

inline std::vector<int64_t> get_shape(const fxfusion::Tensor* tensor) {
    std::vector<int64_t> shape;
    for(auto dim : *tensor->shape()) shape.push_back(dim);
    return shape;
}

inline int64_t get_numel(const std::vector<int64_t>& shape) {
    int64_t numel = 1;
    for (const auto dim : shape) numel *= dim;
    return numel;
}

MemoryManager::MemoryManager(const fxfusion::Graph* graph, const torch::Device& device) {
    TORCH_CHECK(graph != nullptr, "Execution graph is not loaded");

    if (graph->arena_size() > 0) {
        arena_ = torch::empty(
            {static_cast<int64_t>(graph->arena_size())},
            torch::TensorOptions().device(device).dtype(torch::kUInt8)
        );
    }

    registry_.resize(graph->tensors()->size());

    for(const auto* tensor : *graph->tensors()) {
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
            TORCH_CHECK(tensor->offset() >= 0, "Arena-backed tensor has invalid offset");

            size_t offset = static_cast<size_t>(tensor->offset());
            uint8_t* raw_ptr = arena_.data_ptr<uint8_t>();
            
            registry_[id] = torch::from_blob(
                raw_ptr + offset, shape, 
                torch::TensorOptions().device(device).dtype(dtype)
            );

            if (kind == fxfusion::TensorKind_Output) {
                output_ids_.push_back(id);
            }

        } else if (kind == fxfusion::TensorKind_Alias) {

            if (tensor->offset() >= 0) {
                TORCH_CHECK(arena_.defined(), "Arena tensor is not allocated");

                size_t offset = static_cast<size_t>(tensor->offset());
                uint8_t* raw_ptr = arena_.data_ptr<uint8_t>();

                registry_[id] = torch::from_blob(
                    raw_ptr + offset, shape,
                    torch::TensorOptions().device(device).dtype(dtype)
                );
            } else {
                int32_t src_id = tensor->alias_of();
                TORCH_CHECK(src_id >= 0 && src_id < registry_.size(), "Invalid alias source ID");
                TORCH_CHECK(registry_[src_id].defined(), "Alias source tensor is not defined");

                if (registry_[src_id].numel() == get_numel(shape)) {
                    registry_[id] = registry_[src_id].view(shape);
                } else {
                    // Resolve in kernel execution if numel doesn't match, as it might be a narrow alias
                    
                }
            }
            
        } else if (kind == fxfusion::TensorKind_Input) {
            input_ids_.push_back(id);

        } else {
            throw std::runtime_error("Unsupported TensorKind in FlatBuffer");
        }
    }

    outputs_.reserve(output_ids_.size());
    for (auto id : output_ids_) {
        outputs_.push_back(registry_[id]);
    }
}

void MemoryManager::bind_inputs(const std::vector<torch::Tensor>& inputs) {
    TORCH_CHECK(inputs.size() == input_ids_.size(), "Input count mismatch");

    for (size_t i = 0; i < inputs.size(); ++i) {
        registry_[input_ids_[i]] = inputs[i];
    }
}

const std::vector<torch::Tensor>& MemoryManager::get_outputs() const {
    return outputs_;
}

}