#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void linear_relu (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    TORCH_CHECK(false, "linear_relu: not implemented yet");
}

} 