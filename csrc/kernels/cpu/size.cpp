#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void size(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x      = reg[input_ids[0]];
    auto& out          = reg[output_ids[0]];
    const int64_t dim  = params.ints[0];

    out.copy_(torch::tensor(x.size(dim), torch::kInt64));
}

} 