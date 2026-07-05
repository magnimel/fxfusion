#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void layer_norm(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& x      = reg[input_ids[0]];
    const auto& weight = reg[input_ids[1]];
    const auto& bias   = reg[input_ids[2]];
    auto& out          = reg[output_ids[0]];

    const int rank = params.ints[0];
    std::vector<int64_t> normalized_shape(params.ints.begin() + 1, params.ints.begin() + 1 + rank);
    const float eps = params.floats[0];

    out.copy_(torch::layer_norm(x, normalized_shape, weight, bias, eps));
}

} 