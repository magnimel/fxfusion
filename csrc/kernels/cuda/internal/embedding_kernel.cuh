#pragma once

namespace fxfusion::kernels::cuda {

__global__ void embedding_kernel(
    const int64_t* __restrict__ idx, 
    const float* __restrict__ w, 
    float* __restrict__ out, 
    int64_t N, int64_t vocab_size, 
    int64_t d_model
) {

    int64_t k = blockIdx.x;
    int64_t id = idx[k];

    float* out_row = out + k * d_model;
    const float* w_row = w + id * d_model;

    for(int i = threadIdx.x; i < d_model; i += blockDim.x) {
        out_row[i] = w_row[i];
    }

}

} // namespace fxfusion::kernels::cuda