#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void conv2d_relu (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    TORCH_CHECK(false, "conv2d_relu: not implemented yet");
}

} 