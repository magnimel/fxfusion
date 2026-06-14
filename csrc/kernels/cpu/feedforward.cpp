#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void feedforward(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params&) {
    const auto& x   = reg[input_ids[0]];
    const auto& w1  = reg[input_ids[1]];
    const auto& b1  = reg[input_ids[2]];
    const auto& w2  = reg[input_ids[3]];
    const auto& b2  = reg[input_ids[4]];
    auto& out       = reg[output_ids[0]];

    out.copy_(torch::linear(torch::relu(torch::linear(x, w1, b1)), w2, b2));
}

} 