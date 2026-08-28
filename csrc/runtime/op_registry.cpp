#include <functional>
#include <torch/torch.h>
#include <Python.h>
#include <iterator>

#include "op_registry.hpp"

#ifdef USE_CUDA
#include "kernels.cuh"
#include "cache.cuh" 
#endif

#include "kernels.hpp"

namespace fxfusion {

OpRegistry::OpRegistry(const torch::Device& device) {
    const auto num_ops = OpCode_MAX + 1;
    registry_.resize(num_ops);

    if (device.is_cuda()) {
#ifdef USE_CUDA
        register_op(OpCode_Conv2d,            kernels::cuda::conv2d,                kernels::cuda::build_conv2d_cache);
        register_op(OpCode_Conv2dRelu,        kernels::cuda::conv2d_relu,           kernels::cuda::build_conv2d_relu_cache);
        register_op(OpCode_FeedForward,       kernels::cuda::feedforward,           kernels::cuda::build_feedforward_cache);
        register_op(OpCode_Transpose,         kernels::cuda::transpose,             kernels::cuda::build_transpose_cache);
        register_op(OpCode_MHA,               kernels::cuda::mha_flash,             kernels::cuda::build_mha_cache);
        register_op(OpCode_LayerNorm,         kernels::cuda::layer_norm,            kernels::cuda::build_layer_norm_cache);
        register_op(OpCode_AddLayerNorm,      kernels::cuda::add_layer_norm,        kernels::cuda::build_add_layer_norm_cache);
        register_op(OpCode_Linear,            kernels::cuda::linear,                kernels::cuda::build_linear_cache);
        register_op(OpCode_LinearRelu,        kernels::cuda::linear_relu,           kernels::cuda::build_linear_relu_cache);
        register_op(OpCode_Add,               kernels::cuda::add);
        register_op(OpCode_AddRelu,           kernels::cuda::add_relu);
        register_op(OpCode_Relu,              kernels::cuda::relu);
        register_op(OpCode_Mul,               kernels::cuda::mul);
        register_op(OpCode_MaxPool2d,         kernels::cuda::max_pool2d,            kernels::cuda::build_max_pool2d_cache);
        register_op(OpCode_AvgPool2d,         kernels::cuda::avg_pool2d,            kernels::cuda::build_avg_pool2d_cache);
        register_op(OpCode_AdaptiveAvgPool2d, kernels::cuda::adaptive_avg_pool2d,   kernels::cuda::build_adaptive_avg_pool2d_cache);
        register_op(OpCode_Size,              kernels::cuda::size);
        register_op(OpCode_Narrow,            kernels::cuda::narrow);
        register_op(OpCode_Embedding,         kernels::cuda::embedding);
        return; 
#else
        TORCH_CHECK(false, "FXFusion was built without CUDA support"); 
#endif
    }

    register_op(OpCode_Conv2d,            kernels::cpu::conv2d);
    register_op(OpCode_Conv2dRelu,        kernels::cpu::conv2d_relu);
    register_op(OpCode_Linear,            kernels::cpu::linear);
    register_op(OpCode_LinearRelu,        kernels::cpu::linear_relu);
    register_op(OpCode_Add,               kernels::cpu::add);
    register_op(OpCode_AddRelu,           kernels::cpu::add_relu);
    register_op(OpCode_Relu,              kernels::cpu::relu);
    register_op(OpCode_Mul,               kernels::cpu::mul);
    register_op(OpCode_MaxPool2d,         kernels::cpu::max_pool2d);
    register_op(OpCode_AvgPool2d,         kernels::cpu::avg_pool2d);
    register_op(OpCode_AdaptiveAvgPool2d, kernels::cpu::adaptive_avg_pool2d);
    register_op(OpCode_Transpose,         kernels::cpu::transpose);
    register_op(OpCode_Size,              kernels::cpu::size);
    register_op(OpCode_Narrow,            kernels::cpu::narrow);
    register_op(OpCode_Embedding,         kernels::cpu::embedding);
    register_op(OpCode_LayerNorm,         kernels::cpu::layer_norm);
    register_op(OpCode_AddLayerNorm,      kernels::cpu::add_layer_norm);
    register_op(OpCode_MHA,               kernels::cpu::mha);
    register_op(OpCode_FeedForward,       kernels::cpu::feedforward);
}

void OpRegistry::register_op(OpCode op_code, KernelFn kernel, CacheBuilderFn builder) {
    const auto index = static_cast<size_t>(op_code);
    TORCH_CHECK(kernel != nullptr, "Cannot register a null kernel for OpCode index: ", index);
    TORCH_CHECK(index < registry_.size(), "Unsupported OpCode: ", index);
    registry_[index] = OpDef{kernel, builder};
}

const OpDef& OpRegistry::get(OpCode op_code) const {
    const auto index = static_cast<size_t>(op_code);
    TORCH_CHECK(index < registry_.size(), "Unsupported OpCode: ", index);
    TORCH_CHECK(registry_[index].kernel != nullptr, "Unregistered OpCode: ", index);
    return registry_[index];
}

}