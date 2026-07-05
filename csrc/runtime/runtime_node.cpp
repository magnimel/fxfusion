#include <vector>
#include <memory>
#include "runtime_node.hpp"
#include "cache.hpp"

namespace fxfusion {

RuntimeNode::RuntimeNode(const fxfusion::Node* node, KernelFn kernel, TensorRegistry& reg)
    : op_code_(node->op_code())
    , kernel_(kernel)
    , input_ids_(node->input_ids()->begin(), node->input_ids()->end())
    , output_ids_(node->output_ids()->begin(), node->output_ids()->end())
{
    params_.ints.assign(node->int_params()->begin(), node->int_params()->end());
    params_.floats.assign(node->float_params()->begin(), node->float_params()->end());

    build_cache(reg);
}


void RuntimeNode::build_cache(TensorRegistry& reg) {
    switch (op_code_) {
        case OpCode_Transpose:
            cache_ = std::make_unique<TransposeCache>(reg, input_ids_, output_ids_, params_);
            break;
        case OpCode_FeedForward:
            cache_ = std::make_unique<FeedForwardCache>(reg, input_ids_, output_ids_, params_);
            break;
        case OpCode_LayerNorm:
            cache_ = std::make_unique<LayerNormCache>(reg, input_ids_, output_ids_, params_);
            break;
        case OpCode_AddLayerNorm:
            cache_ = std::make_unique<AddLayerNormCache>(reg, input_ids_, output_ids_, params_);
            break;
        case OpCode_MHA:
            cache_ = std::make_unique<MHACache>(reg, input_ids_, output_ids_, params_);
            break;
        default:
            break; 
    }
}

void RuntimeNode::execute(TensorRegistry& reg) {
    kernel_(reg, input_ids_, output_ids_, params_, cache_.get());
}


}

