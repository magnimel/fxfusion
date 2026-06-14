#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void adaptive_avg_pool2d(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x  = reg[input_ids[0]];
    auto& out      = reg[output_ids[0]];
    
    const std::vector<int64_t> output_size = {params.ints[0], params.ints[1]};

    out.copy_(torch::adaptive_avg_pool2d(x, output_size));
}

} 