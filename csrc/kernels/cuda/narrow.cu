#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void narrow(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto&    x      = reg[input_ids[0]];
    const int64_t  dim    = params.ints[0];
    const int64_t  start  = params.ints[1];
    int64_t length        = params.ints[2];

    if (length < 0) length = reg[input_ids[1]].item<int64_t>();
    reg[output_ids[0]].copy_(x.narrow(dim, start, length));
}

} 