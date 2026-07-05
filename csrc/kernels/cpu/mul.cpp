#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void mul(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& x = reg[input_ids[0]];
    auto& out      = reg[output_ids[0]];

    const int mul_mode = params.ints[0];

    switch (mul_mode) {
        case 0: {
            const auto& y = reg[input_ids[1]];
            out.copy_(torch::mul(x, y));
            break;
        }
        case 1:
        case 2:
        case 3:
            out.copy_(torch::mul(x, params.floats[0]));
            break;
        default:
            throw std::runtime_error("mul: unknown mul_mode " + std::to_string(mul_mode));
    }
}

} 