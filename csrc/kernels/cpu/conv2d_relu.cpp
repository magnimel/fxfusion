#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void conv2d_relu(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x   = reg[input_ids[0]];
    const auto& w   = reg[input_ids[1]];
    const auto& b   = reg[input_ids[2]];
    auto& out        = reg[output_ids[0]];

    const std::vector<int64_t> stride   = {params[0], params[1]};
    const std::vector<int64_t> padding  = {params[2], params[3]};
    const std::vector<int64_t> dilation = {params[4], params[5]};
    const int64_t groups                =  params[6];

    out.copy_(torch::relu(torch::conv2d(x, w, b, stride, padding, dilation, groups)));
}

} 