#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"
#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void avg_pool2d(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    const auto& x = reg[input_ids[0]];
    auto& out      = reg[output_ids[0]];

    const std::vector<int64_t> kernel_size = {params[0], params[1]};
    const std::vector<int64_t> stride      = {params[2], params[3]};
    const std::vector<int64_t> padding     = {params[4], params[5]};
    const bool ceil_mode                   =  params[6];

    out.copy_(torch::avg_pool2d(x, kernel_size, stride, padding, ceil_mode));
}

} 