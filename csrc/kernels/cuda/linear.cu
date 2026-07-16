#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

#define TILE_SIZE 32

__global__ void linear_kernel(const float* x, const float* w, const float* b, float* out, int64_t M, int64_t N, int64_t K) {

    __shared__ float xds[TILE_SIZE][TILE_SIZE];
    __shared__ float wds[TILE_SIZE][TILE_SIZE];

    int64_t bx = blockIdx.x;  int64_t by = blockIdx.y;
    int64_t tx = threadIdx.x; int64_t ty = threadIdx.y;

    int64_t row = by * TILE_SIZE + ty;
    int64_t col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int64_t num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int64_t k = 0; k < num_tiles; k++) {
        int64_t x_col = k * TILE_SIZE + tx;
        int64_t w_col = k * TILE_SIZE + ty;

        xds[ty][tx] = (row < M && x_col < K) ? x[row * K + x_col] : 0.0f;
        wds[ty][tx] = (col < N && w_col < K) ? w[col * K + w_col] : 0.0f;
        __syncthreads();

        for (int64_t kT = 0; kT < TILE_SIZE; kT++) {
            sum += xds[ty][kT] * wds[kT][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        out[col + row * N] = sum + b[col];
    }
    
}


__global__ void linear_kernel_ref(const float* x, const float* w, const float* b, float* out, int64_t M, int64_t N, int64_t K) {
    // naive, non-tiled version
    
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < M && col < N) {
        float res = 0.0f;
        for(int64_t k = 0; k < K; k++) {
            res+= x[k + row * K] * w[k + col * K];
        }
        out[col + row * N] = res + b[col];
    }
}

void linear(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache*) {
    const auto& x = reg[input_ids[0]];
    const auto& w = reg[input_ids[1]];
    const auto& b = reg[input_ids[2]];
    auto& out     = reg[output_ids[0]];

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = w.size(0);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    linear_kernel<<<grid, block>>>(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K);
    
}

} 