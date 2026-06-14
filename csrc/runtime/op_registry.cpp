#include <functional>
#include <torch/torch.h>
#include <Python.h>
#include <iterator>

#include "op_registry.hpp"

#ifdef USE_CUDA
#include "kernels.cuh"
#endif

#include "kernels.hpp"

namespace fxfusion {

static KernelSet select_kernels(const torch::Device& device) {
#ifdef USE_CUDA
    if (device.is_cuda()) {
        return {
            kernels::cuda::conv2d,
            kernels::cuda::conv2d_relu,
            kernels::cuda::linear,
            kernels::cuda::linear_relu,
            kernels::cuda::add,
            kernels::cuda::add_relu,
            kernels::cuda::relu,
            kernels::cuda::mul,
            kernels::cuda::max_pool2d,
            kernels::cuda::avg_pool2d,
            kernels::cuda::adaptive_avg_pool2d,
            kernels::cuda::transpose,
            kernels::cuda::size,
            kernels::cuda::narrow,
            kernels::cuda::embedding,
            kernels::cuda::layer_norm,
            kernels::cuda::add_layer_norm,
            kernels::cuda::mha,
            kernels::cuda::feedforward,
        };
    }
#else
    TORCH_CHECK(!device.is_cuda(), "FXFusion was built without CUDA support");
#endif

    return {
        kernels::cpu::conv2d,
        kernels::cpu::conv2d_relu,
        kernels::cpu::linear,
        kernels::cpu::linear_relu,
        kernels::cpu::add,
        kernels::cpu::add_relu,
        kernels::cpu::relu,
        kernels::cpu::mul,
        kernels::cpu::max_pool2d,
        kernels::cpu::avg_pool2d,
        kernels::cpu::adaptive_avg_pool2d,
        kernels::cpu::transpose,
        kernels::cpu::size,
        kernels::cpu::narrow,
        kernels::cpu::embedding,
        kernels::cpu::layer_norm,
        kernels::cpu::add_layer_norm,
        kernels::cpu::mha,
        kernels::cpu::feedforward,
    };
}

OpRegistry::OpRegistry(const torch::Device& device) {
    const auto num_ops = fxfusion::OpCode_MAX + 1;
    registry_.resize(num_ops);

    const auto k = select_kernels(device);

    register_op(OpCode_Conv2d,            k.conv2d);
    register_op(OpCode_Conv2dRelu,        k.conv2d_relu);
    register_op(OpCode_Linear,            k.linear);
    register_op(OpCode_LinearRelu,        k.linear_relu);
    register_op(OpCode_Add,               k.add);
    register_op(OpCode_AddRelu,           k.add_relu);
    register_op(OpCode_Relu,              k.relu);
    register_op(OpCode_Mul,               k.mul);
    register_op(OpCode_MaxPool2d,         k.max_pool2d);
    register_op(OpCode_AvgPool2d,         k.avg_pool2d);
    register_op(OpCode_AdaptiveAvgPool2d, k.adaptive_avg_pool2d);
    register_op(OpCode_Transpose,         k.transpose);
    register_op(OpCode_Size,              k.size);
    register_op(OpCode_Narrow,            k.narrow);
    register_op(OpCode_Embedding,         k.embedding);
    register_op(OpCode_LayerNorm,         k.layer_norm);
    register_op(OpCode_AddLayerNorm,      k.add_layer_norm);
    register_op(OpCode_MHA,               k.mha);
    register_op(OpCode_FeedForward,       k.feedforward);
}

void OpRegistry::register_op(fxfusion::OpCode op_code, KernelFn fn) {
    const auto index = static_cast<size_t>(op_code);
    TORCH_CHECK(index < registry_.size(), "Unsupported OpCode: ", index);
    registry_[index] = std::move(fn);
}

KernelFn OpRegistry::get(fxfusion::OpCode op_code) const {
    const auto index = static_cast<size_t>(op_code);
    TORCH_CHECK(index < registry_.size(), "Unsupported OpCode: ", index);
    TORCH_CHECK(registry_[index] != nullptr, "Unregistered OpCode: ", index);
    return registry_[index];
}

}