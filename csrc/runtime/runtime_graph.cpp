#include "runtime_graph.hpp"
#include "runtime_node.hpp"
#include "op_registry.hpp"
#include "graph_generated.h"

namespace fxfusion {

RuntimeGraph::RuntimeGraph(const fxfusion::Graph* graph, const torch::Device& device) {
    OpRegistry registry(device);  
    nodes_.reserve(graph->nodes()->size());
    for (const auto* node : *graph->nodes()) {
        if (node->op_code() == OpCode_NoOp){
            continue;
        }
        KernelFn kernel = registry.get(node->op_code());
        nodes_.emplace_back(node, kernel);
    }
}

void RuntimeGraph::execute(TensorRegistry& reg) {
    for(auto& node: nodes_) {
        node.execute(reg);
    }
}

}