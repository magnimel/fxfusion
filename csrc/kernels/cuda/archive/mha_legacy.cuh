#pragma once
#include "kernels.cuh"

// =====================================================================
// RETIRED — reference/historical Multi-Head Attention implementations.
//
// This file holds the original multi-kernel (non-fused) MHA pipeline:
// scores -> softmax -> context. Includes both naive and shared-memory
// tiled variants, plus two-pass and single-pass softmax kernels.
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

// =====================================================================
// MULTI-HEAD ATTENTION (LEGACY)
// =====================================================================

// Naive (non-tiled) reference implementation.
//
// Input:  qkv    {batch, seq, qkv_dim} where qkv_dim = 3*d_model
//         mask   {batch, 1, seq, seq} bool
// Output: scores {batch, num_heads, seq, seq}
//
// Launch:
//   dim3 block(16, 16);
//   dim3 grid((seq + block.x - 1) / block.x, (seq + block.y - 1) / block.y, batch * num_heads);
//   scores_kernel_ref<<<grid, block>>>(
//       qkv, mask_ptr, scores, batch, seq, num_heads, head_dim, d_model, qkv_dim, scale_divisor);
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

// Shared-memory tiled version of scores_kernel_ref above.
//
// Input:  qkv    {batch, seq, qkv_dim} where qkv_dim = 3*d_model
//         mask   {batch, 1, seq, seq} bool
// Output: scores {batch, num_heads, seq, seq}
//
// Launch:
//   dim3 block(LINEAR_TILE_SIZE, LINEAR_TILE_SIZE);
//   dim3 grid((seq + block.x - 1) / block.x, (seq + block.y - 1) / block.y, batch * num_heads);
//   scores_kernel<<<grid, block>>>(
//       qkv, mask_ptr, scores, batch, seq, num_heads, head_dim, d_model, qkv_dim, scale_divisor);
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

// Two-pass (max, then exp-sum) in-place softmax over the last dim, one
// block per (b, h, i) row. Matches torch.softmax exactly, including NaN
// on a fully-masked row (no guard) — deliberate, not a bug.
//
// Input/Output: scores {batch, num_heads, seq, seq}
//
// Launch:
//   dim3 block(256);
//   dim3 grid(batch * num_heads * seq);
//   size_t shared_bytes = block.x * sizeof(float);
//   softmax_kernel<<<grid, block, shared_bytes>>>(scores, batch, num_heads, seq);
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

// Single-pass online-softmax variant of softmax_kernel above (per-thread
// streaming max/denominator, combined via a tree-reduction rescale —
// same identity flash_attention_kernel uses across tiles, applied here
// across threads instead).
//
// Input/Output: scores {batch, num_heads, seq, seq}
//
// Launch:
//   dim3 block(256);
//   dim3 grid(batch * num_heads * seq);
//   size_t shared_bytes = 2 * block.x * sizeof(float);  // mT + dT
//   online_softmax_kernel<<<grid, block, shared_bytes>>>(scores, batch, num_heads, seq);
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
        bool has_valid_max = (new_local_max != -INFINITY);
        float rescale = has_valid_max ? expf(local_max - new_local_max) : 0.0f;
        float contribution = has_valid_max ? expf(x - new_local_max) : 0.0f;
        local_dnr = local_dnr * rescale + contribution;
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
            bool has_valid_max = (m_new != -INFINITY);

            float r1 = has_valid_max ? expf(m1 - m_new) : 0.0f;
            float r2 = has_valid_max ? expf(m2 - m_new) : 0.0f;

            dT[tid] = d1 * r1 + d2 * r2;
            mT[tid] = m_new;
        }
        __syncthreads();
    }

    float row_max = mT[0];
    float denom = dT[0];

    // denom == 0 here means every key was masked across the whole row —
    // scores_row[i] becomes 0/0 = NaN, matching torch.softmax's
    // fully-masked-row behavior (deliberate, see softmax_kernel above).
    for (int64_t i = tid; i < seq; i += blockDim.x) {
        scores_row[i] = expf(scores_row[i] - row_max) / denom;
    }
}

// Naive (non-tiled) reference implementation.
//
// Input:  qkv  {batch, seq, qkv_dim} (V slice used, offset by 2*d_model)
//         attn {batch, num_heads, seq, seq} (post-softmax scores)
// Output: ctx  {batch, seq, num_heads, head_dim} == {batch, seq, d_model}
//
// Launch:
//   dim3 block(16, 16);
//   dim3 grid((head_dim + block.x - 1) / block.x, (seq + block.y - 1) / block.y, batch * num_heads);
//   ctx_kernel_ref<<<grid, block>>>(
//       qkv, attn, ctx, batch, seq, num_heads, head_dim, d_model, qkv_dim);
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

// Shared-memory tiled version of ctx_kernel_ref above.
//
// Input:  qkv  {batch, seq, qkv_dim} (V slice used, offset by 2*d_model)
//         attn {batch, num_heads, seq, seq} (post-softmax scores)
// Output: ctx  {batch, seq, num_heads, head_dim} == {batch, seq, d_model}
//              (heads concatenated along the feature axis)
//
// Launch:
//   dim3 block(LINEAR_TILE_SIZE, LINEAR_TILE_SIZE);
//   dim3 grid((head_dim + block.x - 1) / block.x, (seq + block.y - 1) / block.y, batch * num_heads);
//   ctx_kernel<<<grid, block>>>(
//       qkv, attn, ctx, batch, seq, num_heads, head_dim, d_model, qkv_dim);
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

} // namespace fxfusion::kernels::cuda

#endif
