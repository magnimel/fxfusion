#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void layer_norm (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    TORCH_CHECK(false, "layer_norm: not implemented yet");
}

} 