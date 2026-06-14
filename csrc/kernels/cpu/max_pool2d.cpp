#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void max_pool2d(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x = reg[input_ids[0]];
    auto& out      = reg[output_ids[0]];

    const std::vector<int64_t> kernel_size = {params.ints[0], params.ints[1]};
    const std::vector<int64_t> stride      = {params.ints[2], params.ints[3]};
    const std::vector<int64_t> padding     = {params.ints[4], params.ints[5]};
    const std::vector<int64_t> dilation    = {params.ints[6], params.ints[7]};
    const bool ceil_mode                   =  params.ints[8];

    out.copy_(torch::max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode));
}

} 