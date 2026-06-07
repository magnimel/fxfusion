#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void view (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    TORCH_CHECK(false, "view: symbolic, should not be called");
}

} 