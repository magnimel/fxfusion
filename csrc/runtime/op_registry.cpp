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
            kernels::cuda::max_pool2d,
            kernels::cuda::avg_pool2d,
            kernels::cuda::view,
            kernels::cuda::adaptive_avg_pool2d,
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
        kernels::cpu::max_pool2d,
        kernels::cpu::avg_pool2d,
        kernels::cpu::view,
        kernels::cpu::adaptive_avg_pool2d,
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
        register_op(OpCode_MaxPool2d,         k.max_pool2d);
        register_op(OpCode_AvgPool2d,         k.avg_pool2d);
        register_op(OpCode_View,              k.view);
        register_op(OpCode_AdaptiveAvgPool2d, k.adaptive_avg_pool2d);
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