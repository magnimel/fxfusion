#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

// Host-side reference implementation of the transpose index math — kept
// for comparison while developing transpose_kernel. NOT a CUDA kernel
// (no __global__, no launch), and NOT safe to call against device-resident
void transpose_ref(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const TransposeCache*>(cache_base);
    const auto& d = cache->data;

    const auto& x = reg[input_ids[0]];
    auto& out      = reg[output_ids[0]];
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t N = x.numel();

    for (int64_t flat_out = 0; flat_out < N; flat_out++) {
        int64_t flat_in = 0;
        for (int64_t dim = 0; dim < d.dims; dim++) {
            int64_t k = d.perm[dim];
            int64_t idx = (flat_out / d.out_stride[k]) % d.out_shape[k];
            flat_in += idx * d.in_stride[dim];
        }
        out_ptr[flat_out] = x_ptr[flat_in];
    }
}

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

void transpose(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const TransposeCache*>(cache_base);

    const auto& x = reg[input_ids[0]];
    auto& out      = reg[output_ids[0]];
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t N = x.numel();

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    transpose_kernel<<<grid, block>>>(x_ptr, out_ptr, N, cache->data);
}


} 

