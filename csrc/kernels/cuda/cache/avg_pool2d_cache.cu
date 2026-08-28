#include "cache.cuh"
#include "graph_cuda_context.cuh"
#include <cudnn_frontend.h>

namespace fxfusion::kernels::cuda {

namespace fe = cudnn_frontend;

AvgPool2DCache::AvgPool2DCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, bool ADAPTIVE) {
    auto* cuda_ctx = static_cast<GraphCudaContext*>(ctx);
    this->ctx = ctx;

    const auto& x = reg[input_ids[0]];
    auto& out     = reg[output_ids[0]];

    const int64_t N     = x.size(0);
    const int64_t C_in  = x.size(1);
    const int64_t H_in  = x.size(2);
    const int64_t W_in  = x.size(3);

    const int64_t C_out = out.size(1);
    const int64_t H_out = out.size(2);
    const int64_t W_out = out.size(3);

    if (ADAPTIVE) {
        TORCH_CHECK(H_in % H_out == 0 && W_in % W_out == 0,
            "adaptive_avg_pool2d: only evenly-divisible sizes supported, got H_in=",
            H_in, " H_out=", H_out, " W_in=", W_in, " W_out=", W_out);  
    }

    int64_t kH              = ADAPTIVE ? H_in / H_out : params.ints[0];
    int64_t kW              = ADAPTIVE ? W_in / W_out : params.ints[1];
    int64_t stride_h        = ADAPTIVE ? kH : params.ints[2];
    int64_t stride_w        = ADAPTIVE ? kW : params.ints[3];
    int64_t pad_h           = ADAPTIVE ? 0 : params.ints[4];
    int64_t pad_w           = ADAPTIVE ? 0 : params.ints[5];

    graph = std::make_unique<fe::graph::Graph>();

    graph->set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);   

    data.X = graph->tensor(
        fe::graph::Tensor_attributes()
        .set_name("x")
        .set_dim({N, C_in, H_in, W_in})
        .set_stride({C_in * H_in * W_in, H_in * W_in, W_in, 1}));

    data.Y = graph->resample(data.X,
        fe::graph::Resample_attributes()
        .set_resampling_mode(fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
        .set_padding_mode(fe::PaddingMode_t::ZERO_PAD)
        .set_window({kH, kW})
        .set_stride({stride_h, stride_w})
        .set_pre_padding({pad_h, pad_w})
        .set_post_padding({pad_h, pad_w}));

    data.Y->set_output(true)
        .set_dim({N, C_out, H_out, W_out})
        .set_stride({C_out * H_out * W_out, H_out * W_out, W_out, 1});

    TORCH_CHECK(graph->validate().is_good(), "cuDNN graph validation failed");
    TORCH_CHECK(graph->build_operation_graph(cuda_ctx->cudnn_handle()).is_good(), "cuDNN build_operation_graph failed");
    TORCH_CHECK(graph->create_execution_plans({fe::HeurMode_t::A}).is_good(), "cuDNN create_execution_plans failed");
    TORCH_CHECK(graph->check_support().is_good(), "cuDNN check_support failed");
    TORCH_CHECK(graph->build_plans().is_good(), "cuDNN build_plans failed");

    int64_t bytes = 0;
    TORCH_CHECK(graph->get_workspace_size(bytes).is_good(), "cuDNN get_workspace_size failed");
    cuda_ctx->update_workspace_max_size(bytes);
}

AdaptiveAvgPool2DCache::AdaptiveAvgPool2DCache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, bool ADAPTIVE) 
    : AvgPool2DCache(ctx, reg, input_ids, output_ids, params, ADAPTIVE) {
}

std::unique_ptr<Cache> build_avg_pool2d_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<AvgPool2DCache>(ctx, reg, input_ids, output_ids, params, false);
}

std::unique_ptr<Cache> build_adaptive_avg_pool2d_cache(GraphContext* ctx, TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params) {
    return std::make_unique<AdaptiveAvgPool2DCache>(ctx, reg, input_ids, output_ids, params, true);
}

} // namespace fxfusion::kernels::cuda