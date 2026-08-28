#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void adaptive_avg_pool2d (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const AdaptiveAvgPool2DCache*>(cache_base);
    auto* cuda_ctx = static_cast<GraphCudaContext*>(cache->ctx);

    const auto& x  = reg[input_ids[0]];
    auto& out      = reg[output_ids[0]];

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {cache->data.X, x.data_ptr<float>()},
        {cache->data.Y, out.data_ptr<float>()},
    };

    auto status = cache->graph->execute(cuda_ctx->cudnn_handle(), variant_pack, cuda_ctx->workspace_ptr());
    TORCH_CHECK(status.is_good(), "adaptive_avg_pool2d execute failed: ", status.get_message());
}

} // namespace fxfusion::kernels::cuda