#pragma once
#include <vector>
#include <torch/torch.h>
#include "tensor_registry.hpp"

namespace fxfusion::kernels::cuda {

void conv2d              (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void conv2d_relu         (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void linear              (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void linear_relu         (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void add                 (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void add_relu            (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void relu                (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void max_pool2d          (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void avg_pool2d          (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void view                (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);
void adaptive_avg_pool2d (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&);

}