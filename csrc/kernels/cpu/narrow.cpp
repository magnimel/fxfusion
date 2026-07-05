#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void narrow(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto&    x      = reg[input_ids[0]];
    const int64_t  dim    = params.ints[0];
    const int64_t  start  = params.ints[1];
    const int64_t  length = reg[input_ids[1]].item<int64_t>();

    reg[output_ids[0]] = x.narrow(dim, start, length);
}

} 