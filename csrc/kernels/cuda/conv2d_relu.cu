#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

void conv2d_relu(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const Conv2DReluCache*>(cache_base);
    auto* cuda_ctx = static_cast<GraphCudaContext*>(cache->ctx);

    const auto& x = reg[input_ids[0]];
    const auto& w = reg[input_ids[1]];
    const auto& b = reg[input_ids[2]];
    auto& out      = reg[output_ids[0]];

    std::unordered_map<int64_t, void*> variant_pack = {
        {(cache->data).x_uid,   x.data_ptr<float>()},
        {(cache->data).w_uid,   w.data_ptr<float>()},
        {(cache->data).b_uid,   b.data_ptr<float>()},
        {(cache->data).out_uid, out.data_ptr<float>()},
    };

    auto status = cache->graph->execute(cuda_ctx->cudnn_handle(), variant_pack, cuda_ctx->workspace_ptr());
    TORCH_CHECK(status.is_good(), "conv2d execute failed: ", status.get_message());
}

} // namespace fxfusion::kernels::cuda