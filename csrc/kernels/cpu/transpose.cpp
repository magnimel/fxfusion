#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void transpose(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& x  = reg[input_ids[0]];
    auto& out       = reg[output_ids[0]];
    const int64_t dim0 = params.ints[0];
    const int64_t dim1 = params.ints[1];

    out.copy_(torch::transpose(x, dim0, dim1));
}


} 

