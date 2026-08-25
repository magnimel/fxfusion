#pragma once
#include "kernels.cuh"

// =====================================================================
// RETIRED — reference/historical Conv2D implementations only.
//
// This file holds naive/intermediate Conv2D implementations, from
// one-thread-per-tap (ref0) through full weight+input tiling (ref2)
// to the final channel-at-a-time version with grouped-conv and fused
// ReLU support.
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

constexpr int CONV2D_TILE_DIM  = 16;

// =====================================================================
// CONV2D (LEGACY)
// =====================================================================

// ref0: one thread per (kh, kw) kernel tap; each thread reduces over
// C_in for its tap, then a shared-memory tree reduction combines all
// taps into the final output value. No groups support (assumes
// groups=1 — C_in used directly, no group-channel offsetting).
//
// Launch:
//   dim3 block(ksize_w, ksize_h);
//   dim3 grid(H_out * W_out, C_out, N);
//   size_t shared_bytes = ksize_h * ksize_w * sizeof(float);
//   conv2d_kernel_ref0<<<grid, block, shared_bytes>>>(
//       x_ptr, w_ptr, b_ptr, out_ptr,
//       ksize_h, ksize_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
//       groups, C_in, H_in, W_in, C_out, H_out, W_out);
__global__ void conv2d_kernel_ref0(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t ksize_h, int64_t ksize_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h,    int64_t pad_w,
    int64_t dil_h,    int64_t dil_w,
    int64_t groups,
    int64_t C_in,  int64_t H_in,  int64_t W_in,
    int64_t C_out, int64_t H_out, int64_t W_out)
{
    int64_t n     = blockIdx.z;
    int64_t c_out = blockIdx.y;
    int64_t h_out = blockIdx.x / W_out;
    int64_t w_out = blockIdx.x % W_out;

    int64_t kh = threadIdx.y;
    int64_t kw = threadIdx.x;
    int64_t tid = kh * blockDim.x + kw;

    int64_t h_in = h_out * stride_h - pad_h + kh * dil_h;
    int64_t w_in = w_out * stride_w - pad_w + kw * dil_w;

    float partial = 0.0f;

    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
        for (int64_t c_in = 0; c_in < C_in; ++c_in) {
            int64_t x_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
            int64_t w_idx = ((c_out * C_in + c_in) * ksize_h + kh) * ksize_w + kw;
            partial += x[x_idx] * w[w_idx];
        }
    }

    extern __shared__ float smem[];
    smem[tid] = partial;
    __syncthreads();

    int64_t num_threads = blockDim.x * blockDim.y;

    for (int64_t s = 1; s < num_threads; s <<= 1) {
        int64_t index = 2 * s * tid;
        if (index + s < num_threads)
            smem[index] += smem[index + s];
        __syncthreads();
    }

    if (tid == 0) {
        int64_t out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        out[out_idx] = smem[0] + b[c_out];
    }
}

// ref1: caches only the weight slice for c_out in shared memory (full
// C_in worth); input is still read directly from global memory per
// tap. No groups support.
//
// Launch:
//   dim3 block(CONV2D_TILE_DIM, CONV2D_TILE_DIM);
//   dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C_out);
//   size_t shared_bytes = C_in * ksize_h * ksize_w * sizeof(float);
//   conv2d_kernel_ref1<<<grid, block, shared_bytes>>>(
//       x_ptr, w_ptr, b_ptr, out_ptr,
//       ksize_h, ksize_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
//       groups, C_in, H_in, W_in, C_out, H_out, W_out);
__global__ void conv2d_kernel_ref1(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t ksize_h, int64_t ksize_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h,    int64_t pad_w,
    int64_t dil_h,    int64_t dil_w,
    int64_t groups,
    int64_t C_in,  int64_t H_in,  int64_t W_in,
    int64_t C_out, int64_t H_out, int64_t W_out)
{
    extern __shared__ float smem_w[];

    int64_t n     = blockIdx.z / C_out;
    int64_t c_out = blockIdx.z % C_out;

    int64_t h_out = blockIdx.y * CONV2D_TILE_DIM + threadIdx.y;
    int64_t w_out = blockIdx.x * CONV2D_TILE_DIM + threadIdx.x;

    int64_t tid         = threadIdx.y * blockDim.x + threadIdx.x;
    int64_t num_threads = blockDim.x * blockDim.y;
    int64_t slice_size  = C_in * ksize_h * ksize_w;

    const float* w_slice = w + c_out * slice_size;
    for (int64_t i = tid; i < slice_size; i += num_threads)
        smem_w[i] = w_slice[i];
    __syncthreads();

    if (h_out >= H_out || w_out >= W_out)
        return;

    float res = 0.0f;

    for (int64_t c_in = 0; c_in < C_in; ++c_in) {
        for (int64_t kh = 0; kh < ksize_h; ++kh) {
            for (int64_t kw = 0; kw < ksize_w; ++kw) {
                int64_t h_in = h_out * stride_h - pad_h + kh * dil_h;
                int64_t w_in = w_out * stride_w - pad_w + kw * dil_w;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    int64_t x_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int64_t w_idx = (c_in * ksize_h + kh) * ksize_w + kw;
                    res += x[x_idx] * smem_w[w_idx];
                }
            }
        }
    }

    int64_t out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    out[out_idx] = res + b[c_out];
}

// ref2: caches BOTH the full weight slice AND the full multi-channel
// input tile (all C_in channels at once) in shared memory. Correct, but
// shared-memory footprint scales with C_in — can exceed the shared
// memory budget for deep layers (e.g. C_in=256/512), which is why ref3
// (channel-at-a-time, see conv2d_kernel below) replaced it. No groups
// support.
//
// Launch:
//   dim3 block(CONV2D_TILE_DIM, CONV2D_TILE_DIM);
//   dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C_out);
//   int64_t xtsize_h = (CONV2D_TILE_DIM - 1) * stride_h + (ksize_h - 1) * dil_h + 1;
//   int64_t xtsize_w = (CONV2D_TILE_DIM - 1) * stride_w + (ksize_w - 1) * dil_w + 1;
//   size_t shared_bytes = (C_in * ksize_h * ksize_w + C_in * xtsize_h * xtsize_w) * sizeof(float);
//   conv2d_kernel_ref2<<<grid, block, shared_bytes>>>(
//       x_ptr, w_ptr, b_ptr, out_ptr,
//       ksize_h, ksize_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
//       groups, C_in, H_in, W_in, C_out, H_out, W_out);
__global__ void conv2d_kernel_ref2(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t ksize_h, int64_t ksize_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h,    int64_t pad_w,
    int64_t dil_h,    int64_t dil_w,
    int64_t groups,
    int64_t C_in,  int64_t H_in,  int64_t W_in,
    int64_t C_out, int64_t H_out, int64_t W_out)
{
    extern __shared__ float smem[];

    int64_t filter_elems = C_in * ksize_h * ksize_w;
    float* smem_w = smem;
    float* smem_x = smem + filter_elems;

    int64_t n     = blockIdx.z / C_out;
    int64_t c_out = blockIdx.z % C_out;

    int64_t h_out = blockIdx.y * CONV2D_TILE_DIM + threadIdx.y;
    int64_t w_out = blockIdx.x * CONV2D_TILE_DIM + threadIdx.x;

    int64_t tid         = threadIdx.y * blockDim.x + threadIdx.x;
    int64_t num_threads = blockDim.x * blockDim.y;

    const float* w_slice = w + c_out * filter_elems;
    for (int64_t i = tid; i < filter_elems; i += num_threads){
        smem_w[i] = w_slice[i];
    }

    int64_t xtsize_h = (CONV2D_TILE_DIM - 1) * stride_h + (ksize_h - 1) * dil_h + 1;
    int64_t xtsize_w = (CONV2D_TILE_DIM - 1) * stride_w + (ksize_w - 1) * dil_w + 1;
    int64_t xt_elems = C_in * xtsize_h * xtsize_w;

    int64_t h_in_base = blockIdx.y * CONV2D_TILE_DIM * stride_h - pad_h;
    int64_t w_in_base = blockIdx.x * CONV2D_TILE_DIM * stride_w - pad_w;

    for (int64_t i = tid; i < xt_elems; i += num_threads) {
        int64_t c_in =  i / (xtsize_h * xtsize_w);
        int64_t rem  =  i % (xtsize_h * xtsize_w);
        int64_t lh   = rem / xtsize_w;
        int64_t lw   = rem % xtsize_w;

        int64_t h_in = h_in_base + lh;
        int64_t w_in = w_in_base + lw;

        float val = 0.0f;
        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            int64_t x_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
            val = x[x_idx];
        }
        smem_x[i] = val;
    }
    __syncthreads();

    if (h_out >= H_out || w_out >= W_out) return;

    int64_t lh0 = threadIdx.y * stride_h;
    int64_t lw0 = threadIdx.x * stride_w;

    float res = 0.0f;

    for (int64_t c_in = 0; c_in < C_in; ++c_in) {
        for (int64_t kh = 0; kh < ksize_h; ++kh) {
            for (int64_t kw = 0; kw < ksize_w; ++kw) {
                int64_t lh = lh0 + kh * dil_h;
                int64_t lw = lw0 + kw * dil_w;

                int64_t x_idx = (c_in * xtsize_h + lh) * xtsize_w + lw;
                int64_t w_idx = (c_in * ksize_h  + kh) * ksize_w  + kw;

                res += smem_x[x_idx] * smem_w[w_idx];
            }
        }
    }

    int64_t out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    out[out_idx] = res + b[c_out];
}

// conv2d_kernel<RELU>: final/most-optimized version. Channel-at-a-time
// shared-memory tiling (small, fixed footprint regardless of C_in — see
// ref2's comment for why this replaced it), full grouped-conv support,
// and an optional fused ReLU epilogue via the RELU template parameter.
//
// Launch:
//   int64_t xtsize_h = (CONV2D_TILE_DIM - 1) * stride_h + (ksize_h - 1) * dil_h + 1;
//   int64_t xtsize_w = (CONV2D_TILE_DIM - 1) * stride_w + (ksize_w - 1) * dil_w + 1;
//   size_t shared_bytes = (ksize_h * ksize_w + xtsize_h * xtsize_w) * sizeof(float);
//   dim3 block(CONV2D_TILE_DIM, CONV2D_TILE_DIM);
//   dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C_out);
//   conv2d_kernel<RELU><<<grid, block, shared_bytes>>>(
//       x_ptr, w_ptr, b_ptr, out_ptr,
//       ksize_h, ksize_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
//       groups, C_in, H_in, W_in, C_out, H_out, W_out);
template<bool RELU>
__global__ void conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t ksize_h, int64_t ksize_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h,    int64_t pad_w,
    int64_t dil_h,    int64_t dil_w,
    int64_t groups,
    int64_t C_in,  int64_t H_in,  int64_t W_in,
    int64_t C_out, int64_t H_out, int64_t W_out)
{
    extern __shared__ float smem[]; // layout: [input tile | weights (kH*kW)]

    int64_t xtsize_h = (CONV2D_TILE_DIM - 1) * stride_h + (ksize_h - 1) * dil_h + 1;
    int64_t xtsize_w = (CONV2D_TILE_DIM - 1) * stride_w + (ksize_w - 1) * dil_w + 1;

    int64_t xt_elems = xtsize_h * xtsize_w;
    int64_t w_elems = ksize_h * ksize_w;

    float* smem_x = smem;
    float* smem_w = smem + xt_elems;

    int64_t n     = blockIdx.z / C_out;
    int64_t c_out = blockIdx.z % C_out;
    int64_t h_out = blockIdx.y * CONV2D_TILE_DIM + threadIdx.y;
    int64_t w_out = blockIdx.x * CONV2D_TILE_DIM + threadIdx.x;

    int64_t tid         = threadIdx.y * blockDim.x + threadIdx.x;
    int64_t num_threads = blockDim.x * blockDim.y;

    int64_t h_in_base = blockIdx.y * CONV2D_TILE_DIM * stride_h - pad_h;
    int64_t w_in_base = blockIdx.x * CONV2D_TILE_DIM * stride_w - pad_w;

    int64_t C_in_per_group = C_in / groups;
    int64_t C_out_per_group = C_out / groups;

    int64_t g = c_out / C_out_per_group;
    int64_t c_in_start = g * C_in_per_group;

    int64_t w_base_offset = c_out * C_in_per_group * ksize_h * ksize_w;

    float res = 0.0f;

    for (int64_t ic = 0; ic < C_in_per_group; ++ic) {

        int64_t c_in = c_in_start + ic;

        for (int64_t i = tid; i < xt_elems; i += num_threads) {
            int64_t lh = i / xtsize_w;
            int64_t lw = i % xtsize_w;
            int64_t h  = h_in_base + lh;
            int64_t w  = w_in_base + lw;

            float val = 0.0f;
            if (h >= 0 && h < H_in && w >= 0 && w < W_in)
                val = x[((n * C_in + c_in) * H_in + h) * W_in + w];
            smem_x[i] = val;
        }

        const float* w_src = w + w_base_offset + (ic * w_elems);
        for (int64_t i = tid; i < w_elems; i += num_threads)
            smem_w[i] = w_src[i];

        __syncthreads();

        if (h_out < H_out && w_out < W_out) {
            int64_t lh0 = threadIdx.y * stride_h;
            int64_t lw0 = threadIdx.x * stride_w;

            #pragma unroll
            for (int64_t kh = 0; kh < ksize_h; ++kh)
                #pragma unroll
                for (int64_t kw = 0; kw < ksize_w; ++kw) {
                    int64_t lh = lh0 + kh * dil_h;
                    int64_t lw = lw0 + kw * dil_w;
                    res += smem_x[lh * xtsize_w + lw] * smem_w[kh * ksize_w + kw];
                }
        }

        __syncthreads();   // safe for next channel
    }

    if (h_out < H_out && w_out < W_out) {
        int64_t out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
        res = res + b[c_out];
        out[out_idx] = RELU ? fmaxf(0.0f, res) : res;
    }
}

} // namespace fxfusion::kernels::cuda

#endif
