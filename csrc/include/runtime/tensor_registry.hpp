#pragma once
#include <vector>
#include <cstdint>
#include <torch/torch.h>

namespace fxfusion {

using TensorRegistry = std::vector<torch::Tensor>;
using TensorIds      = std::vector<uint32_t>;

struct Params {
    std::vector<int64_t> ints;
    std::vector<float> floats;
};

using KernelFn = void (*)(
    TensorRegistry&,
    const TensorIds&,
    const TensorIds&,
    const Params&
);

}