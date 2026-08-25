#pragma once

namespace fxfusion::kernels::cuda {

__global__ void transpose_kernel(const float* x_ptr, float* out_ptr, int64_t N, TransposeCacheData data) {
    
    for(int64_t flat_out = blockIdx.x * blockDim.x + threadIdx.x;
        flat_out < N;
        flat_out += blockDim.x * gridDim.x
    ) {
        int64_t flat_in = 0;
        for (int64_t dim = 0; dim < data.dims; dim++) {
            int64_t k = data.perm[dim];
            int64_t idx = (flat_out / data.out_stride[k]) % data.out_shape[k];
            flat_in += idx * data.in_stride[dim];
        }
        out_ptr[flat_out] = x_ptr[flat_in];
    }
}

} // namespace fxfusion::kernels::cuda