#pragma once
#include <vector>
#include <torch/torch.h>
#include "graph_generated.h"

namespace fxfusion::kernels::cuda {

void conv2d      (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void conv2d_relu (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void linear      (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void linear_relu (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void add         (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void add_relu    (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void relu        (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void max_pool2d  (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void avg_pool2d  (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void view        (const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);
void adaptive_avg_pool2d(const std::vector<torch::Tensor>& registry, const fxfusion::Node* node);

} 