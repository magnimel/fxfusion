#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void add_layer_norm (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    TORCH_CHECK(false, "add_layer_norm: not implemented yet");
}

} 