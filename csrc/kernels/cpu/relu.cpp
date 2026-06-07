#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void relu(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params&) {
    auto& out = reg[output_ids[0]];
    out.copy_(torch::relu(reg[input_ids[0]]));
}

}