#pragma once
#include "kernels.cuh"

// =====================================================================
// RETIRED — reference/historical Linear (GEMM) implementations only.
//
// This file holds naive/non-tiled and shared-memory tiled GEMM
// implementations, with optional fused ReLU epilogue.
//
// Wrapped in #if 0 to prevent compilation while maintaining IDE syntax
// highlighting. Do not wire these back into OpRegistry without restoring
// their respective memory allocations and cache structures.
//
// Kernels are ordered from earliest/naive to most optimized, matching
// their actual development history. Each kernel has a short launch-example
// comment directly above it.
// =====================================================================

#if 0

namespace fxfusion::kernels::cuda {

constexpr int LINEAR_TILE_SIZE = 32;

// =====================================================================
// LINEAR / GEMM (LEGACY)
// =====================================================================

// naive, non-tiled version — plain per-thread dot product, no shared
// memory, no fused ReLU.
//
// Launch:
//   dim3 block(16, 16);
//   dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
//   linear_kernel_ref<<<grid, block>>>(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K);
__global__ void linear_kernel_ref(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ out,
    int64_t M, int64_t N, int64_t K
) {
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

// naive, non-tiled version, with a fused ReLU epilogue — predates the
// RELU template parameter on linear_kernel below (kept separate rather
// than templated, since this is a standalone reference).
//
// Launch:
//   dim3 block(16, 16);
//   dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
//   linear_relu_kernel_ref<<<grid, block>>>(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K);
__global__ void linear_relu_kernel_ref(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ out,
    int64_t M, int64_t N, int64_t K
) {
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < M && col < N) {
        float res = 0.0f;
        for(int64_t k = 0; k < K; k++) {
            res+= x[k + row * K] * w[k + col * K];
        }
        out[col + row * N] = fmaxf(0.0f, res + b[col]);
    }
}

// Final/most-optimized version: shared-memory tiled GEMM with an
// optional fused ReLU epilogue via the RELU template parameter.
//
// Launch:
//   dim3 block(LINEAR_TILE_SIZE, LINEAR_TILE_SIZE);
//   dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
//   linear_kernel<RELU><<<grid, block>>>(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K);
template<bool RELU>
__global__ void linear_kernel(
    const float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, float* __restrict__ out,
    int64_t M, int64_t N, int64_t K
) {
    __shared__ float xds[LINEAR_TILE_SIZE][LINEAR_TILE_SIZE];
    __shared__ float wds[LINEAR_TILE_SIZE][LINEAR_TILE_SIZE];

    int64_t bx = blockIdx.x;  int64_t by = blockIdx.y;
    int64_t tx = threadIdx.x; int64_t ty = threadIdx.y;

    int64_t row = by * LINEAR_TILE_SIZE + ty;
    int64_t col = bx * LINEAR_TILE_SIZE + tx;

    float res = 0.0f;
    int64_t num_tiles = (K + LINEAR_TILE_SIZE - 1) / LINEAR_TILE_SIZE;

    for (int64_t k = 0; k < num_tiles; k++) {
        int64_t x_col = k * LINEAR_TILE_SIZE + tx;
        int64_t w_col = k * LINEAR_TILE_SIZE + ty;

        xds[ty][tx] = (row < M && x_col < K) ? x[row * K + x_col] : 0.0f;
        wds[ty][tx] = (col < N && w_col < K) ? w[col * K + w_col] : 0.0f;
        __syncthreads();

        for (int64_t kT = 0; kT < LINEAR_TILE_SIZE; kT++) {
            res += xds[ty][kT] * wds[kT][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        res = res + b[col];
        out[col + row * N] = RELU ? fmaxf(0.0f, res) : res;
    }
}

} // namespace fxfusion::kernels::cuda

#endif
