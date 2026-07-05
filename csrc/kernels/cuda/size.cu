#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void size(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& x      = reg[input_ids[0]];
    auto& out          = reg[output_ids[0]];
    const int64_t dim  = params.ints[0];

    out.fill_(x.size(dim));

}

} 