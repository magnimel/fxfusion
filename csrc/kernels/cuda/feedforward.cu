#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void feedforward (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    TORCH_CHECK(false, "feedforward: not implemented yet");
}

} 