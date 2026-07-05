#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void adaptive_avg_pool2d (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    TORCH_CHECK(false, "adaptive_avg_pool2d: not implemented yet");
}

} 