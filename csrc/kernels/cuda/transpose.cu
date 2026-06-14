#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void transpose (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    TORCH_CHECK(false, "transpose: not implemented yet");
}

} 