#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void linear(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params&) {
    const auto& x = reg[input_ids[0]];
    const auto& w = reg[input_ids[1]];
    const auto& b = reg[input_ids[2]];
    auto& out      = reg[output_ids[0]];
    out.copy_(torch::linear(x, w, b));
}

} 