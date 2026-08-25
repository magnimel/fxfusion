#include <vector>
#include <memory>
#include "runtime_node.hpp"

namespace fxfusion {

RuntimeNode::RuntimeNode(GraphContext* ctx, const fxfusion::Node* node, const OpDef& def, TensorRegistry& reg, const torch::Device& device)
    : op_code_(node->op_code())
    , kernel_(def.kernel)
    , input_ids_(node->input_ids()->begin(), node->input_ids()->end())
    , output_ids_(node->output_ids()->begin(), node->output_ids()->end())
{
    params_.ints.assign(node->int_params()->begin(), node->int_params()->end());
    params_.floats.assign(node->float_params()->begin(), node->float_params()->end());

    if (def.cache_builder != nullptr) {
        cache_ = def.cache_builder(ctx, reg, input_ids_, output_ids_, params_);
    }
}

void RuntimeNode::execute(TensorRegistry& reg) {
    kernel_(reg, input_ids_, output_ids_, params_, cache_.get());
}

}