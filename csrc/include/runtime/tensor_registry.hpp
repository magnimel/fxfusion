#pragma once
#include <vector>
#include <torch/torch.h>

namespace fxfusion {

using TensorRegistry = std::vector<torch::Tensor>;
using TensorIds      = std::vector<uint32_t>;
using Params         = std::vector<int32_t>;

using KernelFn = void (*)(
    TensorRegistry&,
    const TensorIds&,
    const TensorIds&,
    const Params&
);

}