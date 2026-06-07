#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void add_relu (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    TORCH_CHECK(false, "add_relu: not implemented yet");
}

} 