#pragma once
#include "kernels.cuh"

// =====================================================================
// RETIRED — reference/historical implementation only.
//
// This file holds the original multi-kernel (non-fused) MHA pipeline
// (scores -> softmax -> context) and naive/intermediate Conv2D implementations.
// 
// Wrapped in #if 0 to prevent compilation while maintaining IDE syntax highlighting.
// Do not wire these back into OpRegistry without restoring their respective
// memory allocations and cache structures.
// =====================================================================

#if 0 

namespace fxfusion::kernels::cuda {

// =====================================================================
// MULTI-HEAD ATTENTION (LEGACY)
// =====================================================================

// Naive (non-tiled) reference implementation
// Input:  qkv    {batch, seq, qkv_dim} where qkv_dim = 3*d_model
//         mask   {batch, 1, seq, seq} bool
// Output: scores {batch, num_heads, seq, seq}
__global__ void scores_kernel_ref(const float* qkv, const bool* mask, float* scores,
                               int64_t batch, int64_t seq, int64_t num_heads,
                               int64_t head_dim, int64_t d_model, int64_t qkv_dim,
                               float scale_divisor) 
{
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t bh = blockIdx.z;
    int64_t b = bh / num_heads;
    int64_t h = bh % num_heads;

    if (i >= seq || j >= seq) return;

    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) { 
        int64_t q_offset = (b * seq * qkv_dim) + (i * qkv_dim) + h * head_dim + d;
        int64_t k_offset = (b * seq * qkv_dim) + (j * qkv_dim) + d_model + h * head_dim + d;   
        sum += qkv[q_offset] * qkv[k_offset];
    }   

    bool keep = mask[b * seq * seq + i * seq + j];
    int64_t out_idx = b * num_heads * seq * seq + h * seq * seq + i * seq + j;
    scores[out_idx] = keep ? (sum / scale_divisor) : -INFINITY;
}

// Input:  qkv    {batch, seq, qkv_dim} where qkv_dim = 3*d_model
//         mask   {batch, 1, seq, seq} bool
// Output: scores {batch, num_heads, seq, seq}
__global__ void scores_kernel(const float* qkv, const bool* mask, float* scores,
                               int64_t batch, int64_t seq, int64_t num_heads,
                               int64_t head_dim, int64_t d_model, int64_t qkv_dim,
                               float scale_divisor) 
{
    __shared__ float qds[LINEAR_TILE_SIZE][LINEAR_TILE_SIZE];
    __shared__ float kds[LINEAR_TILE_SIZE][LINEAR_TILE_SIZE];

    int64_t bx = blockIdx.x;  int64_t by = blockIdx.y;
    int64_t tx = threadIdx.x; int64_t ty = threadIdx.y;

    int64_t i = by * LINEAR_TILE_SIZE + ty;
    int64_t j = bx * LINEAR_TILE_SIZE + tx;

    int64_t bh = blockIdx.z;
    int64_t b = bh / num_heads;
    int64_t h = bh % num_heads;

    float sum = 0.0f;

    int64_t num_tiles = (head_dim + LINEAR_TILE_SIZE - 1) / LINEAR_TILE_SIZE;

    for (int64_t k = 0; k < num_tiles; k++) {
        int64_t q_col = k * LINEAR_TILE_SIZE + tx;
        int64_t k_col = k * LINEAR_TILE_SIZE + ty;

        int64_t q_offset = (b * seq * qkv_dim) + (i * qkv_dim) + h * head_dim + q_col;
        int64_t k_offset = (b * seq * qkv_dim) + (j * qkv_dim) + d_model + h * head_dim + k_col;  
        
        qds[ty][tx] = (i < seq && q_col < head_dim) ? qkv[q_offset] : 0.0f;
        kds[ty][tx] = (j < seq && k_col < head_dim) ? qkv[k_offset] : 0.0f;

        __syncthreads();
        for (int64_t kT = 0; kT < LINEAR_TILE_SIZE; kT++) {
            sum += qds[ty][kT] * kds[kT][tx];
        }
        __syncthreads();
    }   

    if (i < seq && j < seq) {
        bool keep = mask[b * seq * seq + i * seq + j];
        int64_t out_idx = b * num_heads * seq * seq + h * seq * seq + i * seq + j;
        scores[out_idx] = keep ? (sum / scale_divisor) : -INFINITY;
    }
}

// Input/Output: scores {batch, num_heads, seq, seq}
// Two-pass (max, then exp-sum) — matches torch.softmax exactly
__global__ void softmax_kernel(float* scores, int64_t batch, int64_t num_heads, int64_t seq) {
    extern __shared__ float tmp[];

    int64_t row = blockIdx.x;

    int64_t b = row / (num_heads * seq);
    int64_t h = (row / seq) % num_heads;
    int64_t s = row % seq;

    int64_t offset = b * num_heads * seq * seq + h * seq * seq + s * seq;
    float* scores_row = scores + offset;

    int64_t tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int64_t i = tid; i < seq; i += blockDim.x) {
        local_max = fmaxf(local_max, scores_row[i]);
    }
    tmp[tid] = local_max;
    __syncthreads();
    for (int64_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) tmp[tid] = fmaxf(tmp[tid], tmp[tid + stride]);
        __syncthreads();
    }
    __shared__ float row_max;
    if (tid == 0) row_max = tmp[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int64_t i = tid; i < seq; i += blockDim.x) {
        float e = expf(scores_row[i] - row_max);
        scores_row[i] = e;
        local_sum += e;
    }
    tmp[tid] = local_sum;
    __syncthreads();
    for (int64_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) tmp[tid] += tmp[tid + stride];
        __syncthreads();
    }
    __shared__ float denom;
    if (tid == 0) denom = tmp[0];
    __syncthreads();

    for (int64_t i = tid; i < seq; i += blockDim.x) {
        scores_row[i] = scores_row[i] / denom;
    }
}

// Single-pass online-softmax variant of softmax_kernel above.
__global__ void online_softmax_kernel(float* scores, int64_t batch, int64_t num_heads, int64_t seq) {
    extern __shared__ float smem[];
    float* mT = smem;
    float* dT = smem + blockDim.x;

    int64_t row = blockIdx.x;

    int64_t b = row / (num_heads * seq);
    int64_t h = (row / seq) % num_heads;
    int64_t s = row % seq;

    int64_t offset = b * num_heads * seq * seq + h * seq * seq + s * seq;
    float* scores_row = scores + offset;

    int64_t tid = threadIdx.x;

    float local_max = -INFINITY;
    float local_dnr = 0.0f;
    for (int64_t i = tid; i < seq; i += blockDim.x) {
        float x = scores_row[i];
        float new_local_max = fmaxf(local_max, x);
        local_dnr = local_dnr * expf(local_max - new_local_max) + expf(x - new_local_max);
        local_max = new_local_max;
    }
    mT[tid] = local_max;
    dT[tid] = local_dnr;
    __syncthreads();

    for (int64_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float m1 = mT[tid];
            float d1 = dT[tid];
            float m2 = mT[tid + stride];
            float d2 = dT[tid + stride];
            
            float m_new = fmaxf(m1, m2);
            
            dT[tid] = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
            mT[tid] = m_new;
        }
        __syncthreads();
    }

    float row_max = mT[0];
    float denom = dT[0];

    for (int64_t i = tid; i < seq; i += blockDim.x) {
        scores_row[i] = expf(scores_row[i] - row_max) / denom;
    }
}

// Naive (non-tiled) reference implementation 
// Output: ctx  {batch, seq, num_heads, head_dim} == {batch, seq, d_model}
__global__ void ctx_kernel_ref(const float* qkv, const float* attn, float* ctx,
                                int64_t batch, int64_t seq, int64_t num_heads,
                                int64_t head_dim, int64_t d_model, int64_t qkv_dim)
{
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t bh = blockIdx.z;
    int64_t b = bh / num_heads;
    int64_t h = bh % num_heads;

    if (i >= seq || j >= head_dim) return;

    float sum = 0.0f;
    for (int64_t k = 0; k < seq; k++) {
        int64_t a_offset = (b * num_heads * seq * seq) + (h * seq * seq) + i * seq + k;
        int64_t v_offset = (b * seq * qkv_dim) + (k * qkv_dim) + 2 * d_model + h * head_dim + j;
        sum += attn[a_offset] * qkv[v_offset];
    }

    int64_t out_idx = b * seq * num_heads * head_dim + i * num_heads * head_dim + h * head_dim + j;
    ctx[out_idx] = sum;
}

// Tiled ctx kernel
__global__ void ctx_kernel(const float* qkv, float* attn, float* ctx,
                            int64_t batch, int64_t seq, int64_t num_heads,
                            int64_t head_dim, int64_t d_model, int64_t qkv_dim)
{
    __shared__ float ads[LINEAR_TILE_SIZE][LINEAR_TILE_SIZE];
    __shared__ float vds[LINEAR_TILE_SIZE][LINEAR_TILE_SIZE];

    int64_t bx = blockIdx.x;  int64_t by = blockIdx.y;
    int64_t tx = threadIdx.x; int64_t ty = threadIdx.y;

    int64_t i = by * LINEAR_TILE_SIZE + ty;
    int64_t j = bx * LINEAR_TILE_SIZE + tx;

    int64_t bh = blockIdx.z;
    int64_t b = bh / num_heads;
    int64_t h = bh % num_heads;

    float sum = 0.0f;
    int64_t num_tiles = (seq + LINEAR_TILE_SIZE - 1) / LINEAR_TILE_SIZE;

    for (int64_t k = 0; k < num_tiles; k++) {
        int64_t a_col = k * LINEAR_TILE_SIZE + tx;
        int64_t v_col = k * LINEAR_TILE_SIZE + ty;

        int64_t a_offset = (b * num_heads * seq * seq) + (h * seq * seq) + i * seq + a_col;
        int64_t v_offset = (b * seq * qkv_dim) + (v_col * qkv_dim) + 2 * d_model + h * head_dim + j;

        ads[ty][tx] = (i < seq && a_col < seq) ? attn[a_offset] : 0.0f;
        vds[ty][tx] = (j < head_dim && v_col < seq) ? qkv[v_offset] : 0.0f;

        __syncthreads();
        for (int64_t kT = 0; kT < LINEAR_TILE_SIZE; kT++) {
            sum += ads[ty][kT] * vds[kT][tx];
        }
        __syncthreads();
    }

    if (i < seq && j < head_dim) {
        int64_t out_idx = b * seq * num_heads * head_dim + i * num_heads * head_dim + h * head_dim + j;
        ctx[out_idx] = sum;
    }
}

// =====================================================================
// CONVOLUTION (LEGACY)
// =====================================================================

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

} 

#endif 