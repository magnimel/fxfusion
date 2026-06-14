#include <vector>
#include "runtime_node.hpp"

namespace fxfusion {

RuntimeNode::RuntimeNode(const fxfusion::Node* node, KernelFn kernel)
    : op_code_(node->op_code())
    , kernel_(kernel)
    , input_ids_(node->input_ids()->begin(), node->input_ids()->end())
    , output_ids_(node->output_ids()->begin(), node->output_ids()->end())
{
    params_.ints.assign(node->int_params()->begin(), node->int_params()->end());
    params_.floats.assign(node->float_params()->begin(), node->float_params()->end());
}

void RuntimeNode::execute(TensorRegistry& reg) {
    kernel_(reg, input_ids_, output_ids_, params_);
}


}

