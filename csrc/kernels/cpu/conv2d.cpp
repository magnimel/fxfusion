#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void conv2d(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& x   = reg[input_ids[0]];
    const auto& w   = reg[input_ids[1]];
    const auto& b   = reg[input_ids[2]];
    auto& out        = reg[output_ids[0]];

    const std::vector<int64_t> stride   = {params.ints[0], params.ints[1]};
    const std::vector<int64_t> padding  = {params.ints[2], params.ints[3]};
    const std::vector<int64_t> dilation = {params.ints[4], params.ints[5]};
    const int64_t groups                =  params.ints[6];

    out.copy_(torch::conv2d(x, w, b, stride, padding, dilation, groups));
}

} 