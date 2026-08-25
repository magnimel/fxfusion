#include "runtime_graph.hpp"
#include "runtime_node.hpp"
#include "op_registry.hpp"
#include "graph_generated.h"

#ifdef USE_CUDA
#include "graph_cuda_context.cuh"
#endif

namespace fxfusion {

RuntimeGraph::RuntimeGraph(const fxfusion::Graph* graph, TensorRegistry& reg, const torch::Device& device) {
    if (device.is_cuda()) {
#ifdef USE_CUDA
        context_ = std::make_unique<kernels::cuda::GraphCudaContext>();
#else
        TORCH_CHECK(false, "FXFusion was built without CUDA support"); 
#endif
    }

    OpRegistry registry(device);
    nodes_.reserve(graph->nodes()->size());
    for (const auto* node : *graph->nodes()) {
        if (node->op_code() == OpCode_NoOp) {
            continue;
        }
        const OpDef& def = registry.get(node->op_code());
        nodes_.emplace_back(context_.get(), node, def, reg, device);
    }

    if (context_) { context_->ensure_workspace(); }
}

void RuntimeGraph::execute(TensorRegistry& reg) {
    for (auto& node : nodes_) {
        node.execute(reg);
    }
}

} 