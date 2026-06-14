#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void add_relu(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params&) {
    const auto& a = reg[input_ids[0]];
    const auto& b = reg[input_ids[1]];
    auto& out      = reg[output_ids[0]];
    
    out.copy_(torch::relu(torch::add(a, b)));
}

} 