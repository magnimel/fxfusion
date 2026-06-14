#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void embedding(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params&) {
    const auto& indices = reg[input_ids[0]];
    const auto& weight  = reg[input_ids[1]];
    auto& out           = reg[output_ids[0]];

    out.copy_(torch::embedding(weight, indices));
}

} 