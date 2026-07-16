#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

#define TILE_SIZE 32

// Naive (non-tiled) reference implementation — NOT used in mha()'s dispatch,
// kept for comparison/debugging only.
//
// Input:  qkv    {batch, seq, qkv_dim} where qkv_dim = 3*d_model
//         mask   {batch, seq, seq} bool
// Output: scores {batch, num_heads, seq, seq}
__global__ void scores_kernel_ref(const float* qkv, const bool* mask, float* scores,
                               int64_t batch, int64_t seq, int64_t num_heads,
                               int64_t head_dim, int64_t d_model, int64_t qkv_dim,
                               float scale_divisor) 
{
    // qkv: {batch, seq, 3*d_model} q: {batch, num_heads, seq, head_dim}
    // qkv: {batch, seq, 3, num_heads, head_dim}
    // qkv: (b * seq * qkv_dim) + (s * qkv_dim) + {0,1,2} * d_model + h * head_dim + d 
    int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t bh = blockIdx.z;
    int64_t b = bh / num_heads;
    int64_t h = bh % num_heads;

    if (i >= seq || j >= seq) return;

    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) { // was `k++` — undefined variable, fixed to `d++`
        int64_t q_offset = (b * seq * qkv_dim) + (i * qkv_dim) + h * head_dim + d;
        int64_t k_offset = (b * seq * qkv_dim) + (j * qkv_dim) + d_model + h * head_dim + d;   
        sum += qkv[q_offset] * qkv[k_offset];
    }   

    bool keep = mask[b * seq * seq + i * seq + j];
    int64_t out_idx = b * num_heads * seq * seq + h * seq * seq + i * seq + j;
    scores[out_idx] = keep ? (sum / scale_divisor) : -INFINITY;
}

// Tiled scores kernel — used in mha()'s dispatch.
//
// Input:  qkv    {batch, seq, qkv_dim} where qkv_dim = 3*d_model
//         mask   {batch, seq, seq} bool
// Output: scores {batch, num_heads, seq, seq}
__global__ void scores_kernel(const float* qkv, const bool* mask, float* scores,
                               int64_t batch, int64_t seq, int64_t num_heads,
                               int64_t head_dim, int64_t d_model, int64_t qkv_dim,
                               float scale_divisor) 
{
    // qkv: (b * seq * qkv_dim) + (s * qkv_dim) + {0,1,2} * d_model + h * head_dim + d 
    __shared__ float qds[TILE_SIZE][TILE_SIZE];
    __shared__ float kds[TILE_SIZE][TILE_SIZE];

    int64_t bx = blockIdx.x;  int64_t by = blockIdx.y;
    int64_t tx = threadIdx.x; int64_t ty = threadIdx.y;

    int64_t i = by * TILE_SIZE + ty;
    int64_t j = bx * TILE_SIZE + tx;

    int64_t bh = blockIdx.z;
    int64_t b = bh / num_heads;
    int64_t h = bh % num_heads;

    float sum = 0.0f;

    int64_t num_tiles = (head_dim + TILE_SIZE - 1) / TILE_SIZE;

    for (int64_t k = 0; k < num_tiles; k++) {
        int64_t q_col = k * TILE_SIZE + tx;
        int64_t k_col = k * TILE_SIZE + ty;

        int64_t q_offset = (b * seq * qkv_dim) + (i * qkv_dim) + h * head_dim + q_col;
        int64_t k_offset = (b * seq * qkv_dim) + (j * qkv_dim) + d_model + h * head_dim + k_col;  
        
        qds[ty][tx] = (i < seq && q_col < head_dim) ? qkv[q_offset] : 0.0f;
        kds[ty][tx] = (j < seq && k_col < head_dim) ? qkv[k_offset] : 0.0f;

        __syncthreads();
        for (int64_t kT = 0; kT < TILE_SIZE; kT++) {
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

// Input/Output: scores {batch, num_heads, seq, seq} — softmax applied
// in-place over the last dim (key positions), one block per (b,h,i) row.
// NOTE: matches torch.softmax exactly, including NaN on a fully-masked
// row (no guard) — deliberate choice, not a bug.
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

// Naive (non-tiled) reference implementation — NOT used in mha()'s dispatch,
// kept for comparison/debugging only.
//
// Input:  qkv  {batch, seq, qkv_dim} (V slice used, offset by 2*d_model)
//         attn {batch, num_heads, seq, seq} (post-softmax scores)
// Output: ctx  {batch, seq, num_heads, head_dim} == {batch, seq, d_model}
//              (heads concatenated along the feature axis)
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

// Tiled ctx kernel — used in mha()'s dispatch.
//
// Input:  qkv  {batch, seq, qkv_dim} (V slice used, offset by 2*d_model)
//         attn {batch, num_heads, seq, seq} (post-softmax scores)
// Output: ctx  {batch, seq, num_heads, head_dim} == {batch, seq, d_model}
//              (heads concatenated along the feature axis)
__global__ void ctx_kernel(const float* qkv, float* attn, float* ctx,
                            int64_t batch, int64_t seq, int64_t num_heads,
                            int64_t head_dim, int64_t d_model, int64_t qkv_dim)
{
    __shared__ float ads[TILE_SIZE][TILE_SIZE];
    __shared__ float vds[TILE_SIZE][TILE_SIZE];

    int64_t bx = blockIdx.x;  int64_t by = blockIdx.y;
    int64_t tx = threadIdx.x; int64_t ty = threadIdx.y;

    int64_t i = by * TILE_SIZE + ty;
    int64_t j = bx * TILE_SIZE + tx;

    int64_t bh = blockIdx.z;
    int64_t b = bh / num_heads;
    int64_t h = bh % num_heads;

    float sum = 0.0f;
    int64_t num_tiles = (seq + TILE_SIZE - 1) / TILE_SIZE;

    for (int64_t k = 0; k < num_tiles; k++) {
        int64_t a_col = k * TILE_SIZE + tx;
        int64_t v_col = k * TILE_SIZE + ty;

        int64_t a_offset = (b * num_heads * seq * seq) + (h * seq * seq) + i * seq + a_col;
        int64_t v_offset = (b * seq * qkv_dim) + (v_col * qkv_dim) + 2 * d_model + h * head_dim + j;

        ads[ty][tx] = (i < seq && a_col < seq) ? attn[a_offset] : 0.0f;
        vds[ty][tx] = (j < head_dim && v_col < seq) ? qkv[v_offset] : 0.0f;

        __syncthreads();
        for (int64_t kT = 0; kT < TILE_SIZE; kT++) {
            sum += ads[ty][kT] * vds[kT][tx];
        }
        __syncthreads();
    }

    if (i < seq && j < head_dim) {
        int64_t out_idx = b * seq * num_heads * head_dim + i * num_heads * head_dim + h * head_dim + j;
        ctx[out_idx] = sum;
    }
}


void mha(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto* cache = static_cast<const MHACache*>(cache_base);

    const auto& x               = reg[input_ids[0]];
    const auto& mask            = reg[input_ids[1]];
    const auto& qkv_weight      = reg[input_ids[2]];
    const auto& qkv_bias        = reg[input_ids[3]];
    const auto& out_weight      = reg[input_ids[4]];
    const auto& out_bias        = reg[input_ids[5]];
    auto& out                   = reg[output_ids[0]];

    const int64_t num_heads     = params.ints[0];
    const int64_t head_dim      = params.ints[1];
    const int64_t d_model       = params.ints[2];
    const int64_t qkv_dim       = params.ints[3];
    const float   scale_divisor = params.floats[0];

    const float* x_ptr          = x.data_ptr<float>();
    const bool*  mask_ptr       = mask.data_ptr<bool>();
    const float* qkv_w_ptr      = qkv_weight.data_ptr<float>();
    const float* qkv_b_ptr      = qkv_bias.data_ptr<float>();
    const float* out_w_ptr      = out_weight.data_ptr<float>();
    const float* out_b_ptr      = out_bias.data_ptr<float>();
    float* out_ptr              = out.data_ptr<float>();

    float* qkv     = (cache->data).qkv;
    float* scores  = (cache->data).scores;
    float* ctx     = (cache->data).ctx;
    int64_t batch  = (cache->data).batch;
    int64_t seq    = (cache->data).seq;

    // --- QKV projection ---
    {
        int64_t K = d_model;
        int64_t M = batch * seq;
        int64_t N = qkv_dim;
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        linear_kernel<<<grid, block>>>(x_ptr, qkv_w_ptr, qkv_b_ptr, qkv, M, N, K);
    }

    // --- Scores ---
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((seq + block.x - 1) / block.x, (seq + block.y - 1) / block.y, batch * num_heads);
        scores_kernel<<<grid, block>>>(qkv, mask_ptr, scores, batch, seq, num_heads, head_dim, d_model, qkv_dim, scale_divisor);
    }

    // --- Softmax ---
    {
        dim3 block(256);
        dim3 grid(batch * num_heads * seq);
        size_t shared_bytes = block.x * sizeof(float);
        softmax_kernel<<<grid, block, shared_bytes>>>(scores, batch, num_heads, seq);
    }

    // --- Ctx ---
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((head_dim + block.x - 1) / block.x, (seq + block.y - 1) / block.y, batch * num_heads);
        ctx_kernel<<<grid, block>>>(qkv, scores, ctx, batch, seq, num_heads, head_dim, d_model, qkv_dim);
    }

    // --- Out projection ---
    {
        int64_t K = d_model;
        int64_t M = batch * seq;
        int64_t N = d_model;
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        linear_kernel<<<grid, block>>>(ctx, out_w_ptr, out_b_ptr, out_ptr, M, N, K);
    }
}

}