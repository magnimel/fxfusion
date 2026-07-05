#pragma once
#include <vector>
#include <torch/torch.h>
#include "tensor_registry.hpp"
#include "cache.hpp"

namespace fxfusion::kernels::cpu {

void conv2d              (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void conv2d_relu         (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void linear              (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void linear_relu         (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void add                 (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void add_relu            (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void relu                (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void mul                 (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void max_pool2d          (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void avg_pool2d          (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void adaptive_avg_pool2d (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void transpose           (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void size                (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void narrow              (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void embedding           (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void layer_norm          (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void add_layer_norm      (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void mha                 (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);
void feedforward         (TensorRegistry&, const TensorIds&, const TensorIds&, const Params&, const Cache*);

}