#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

__global__ void flash_attention_kernel(
    const float* __restrict__ qkv,
    const bool* __restrict__ mask,
    float* __restrict__ ctx,
    int64_t batch, int64_t seq, int64_t num_heads,
    int64_t head_dim, int64_t d_model, int64_t qkv_dim,
    float scale_divisor
) {
    __shared__ float Q_s[FLASH_BLOCK_SIZE][FLASH_BLOCK_SIZE];
    __shared__ float K_s[FLASH_BLOCK_SIZE][FLASH_BLOCK_SIZE];
    __shared__ float V_s[FLASH_BLOCK_SIZE][FLASH_BLOCK_SIZE];
    __shared__ float S_s[FLASH_BLOCK_SIZE][FLASH_BLOCK_SIZE];  
    __shared__ float P_s[FLASH_BLOCK_SIZE][FLASH_BLOCK_SIZE];  
    
    int64_t bx = blockIdx.x;
    int64_t by = blockIdx.y;
    int64_t bh = blockIdx.z;
    int64_t b = bh / num_heads;
    int64_t h = bh % num_heads;

    int64_t tx = threadIdx.x;
    int64_t ty = threadIdx.y;

    int64_t q_row  = by * FLASH_BLOCK_SIZE + ty;
    int64_t out_col = bx * FLASH_BLOCK_SIZE + tx;

    float O_acc = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    int64_t num_seq_tiles  = (seq + FLASH_BLOCK_SIZE - 1) / FLASH_BLOCK_SIZE;
    int64_t num_head_tiles = (head_dim + FLASH_BLOCK_SIZE - 1) / FLASH_BLOCK_SIZE;

    for (int64_t k_step = 0; k_step < num_seq_tiles; k_step++) {
        int64_t kv_row = k_step * FLASH_BLOCK_SIZE + ty;
        int64_t kv_col = k_step * FLASH_BLOCK_SIZE + tx;

        float score = 0.0f;
        for (int64_t d_step = 0; d_step < num_head_tiles; d_step++) {
            int64_t q_d    = d_step * FLASH_BLOCK_SIZE + tx;
            int64_t k_d    = d_step * FLASH_BLOCK_SIZE + ty;

            int64_t q_offset = (b * seq * qkv_dim) + (q_row * qkv_dim)  + 0 * d_model + h * head_dim + q_d;
            int64_t k_offset = (b * seq * qkv_dim) + (kv_col * qkv_dim) + 1 * d_model + h * head_dim + k_d;

            Q_s[ty][tx] = (q_row < seq && q_d < head_dim) ? qkv[q_offset] : 0.0f;
            K_s[ty][tx] = (kv_col < seq && k_d < head_dim) ? qkv[k_offset] : 0.0f;
            __syncthreads();

            for (int d = 0; d < FLASH_BLOCK_SIZE; d++) {
                score += Q_s[ty][d] * K_s[d][tx];
            }
            __syncthreads();
        }

        if (q_row < seq && kv_col < seq) {
            bool keep = mask[b * seq * seq + q_row * seq + kv_col];
            S_s[ty][tx] = keep ? (score / scale_divisor) : -INFINITY;
        } else {
            S_s[ty][tx] = -INFINITY;
        }
        __syncthreads();

        float local_max = -INFINITY;
        for (int i = 0; i < FLASH_BLOCK_SIZE; i++) {
            local_max = fmaxf(local_max, S_s[ty][i]);
        }

        float new_max = fmaxf(running_max, local_max);
        bool nothing_seen_yet = (new_max == -INFINITY);

        float scale = nothing_seen_yet ? 0.0f : expf(running_max - new_max);
        O_acc *= scale;

        float local_sum = 0.0f;
        for (int i = 0; i < FLASH_BLOCK_SIZE; i++) {
            float exp_val = nothing_seen_yet ? 0.0f : expf(S_s[ty][i] - new_max);
            P_s[ty][i] = exp_val;   
            local_sum += exp_val;
        }
        running_sum = running_sum * scale + local_sum;
        running_max = new_max;
        __syncthreads();

        int64_t v_offset = (b * seq * qkv_dim) + (kv_row * qkv_dim) + 2 * d_model + h * head_dim + out_col;
        V_s[ty][tx] = (kv_row < seq && out_col < head_dim) ? qkv[v_offset] : 0.0f;
        __syncthreads();

        for (int k = 0; k < FLASH_BLOCK_SIZE; k++) {
            O_acc += P_s[ty][k] * V_s[k][tx];   
        }
        __syncthreads();
    }

    if (q_row < seq && out_col < head_dim) {
        int64_t out_idx = b * seq * d_model + q_row * d_model + h * head_dim + out_col;
        ctx[out_idx] = O_acc / running_sum;
    }
}

void mha_flash(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
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
    float* ctx     = (cache->data).ctx;
    int64_t batch  = (cache->data).batch;
    int64_t seq    = (cache->data).seq;

    // --- QKV Projection ---
    {
        int64_t K = d_model;
        int64_t M = batch * seq;
        int64_t N = qkv_dim;
        dim3 block(LINEAR_TILE_SIZE, LINEAR_TILE_SIZE);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        linear_kernel<<<grid, block>>>(x_ptr, qkv_w_ptr, qkv_b_ptr, qkv, M, N, K);
    }

    // --- Flash Attention ---
    {
        dim3 block(FLASH_BLOCK_SIZE, FLASH_BLOCK_SIZE);
        dim3 grid((head_dim + block.x - 1) / block.x, (seq + block.y - 1) / block.y, batch * num_heads);
        flash_attention_kernel<<<grid, block>>>(qkv, mask_ptr, ctx, batch, seq, num_heads, head_dim, d_model, qkv_dim, scale_divisor);
    }

    // --- Out Projection ---
    {
        int64_t K = d_model;
        int64_t M = batch * seq;
        int64_t N = d_model;
        dim3 block(LINEAR_TILE_SIZE, LINEAR_TILE_SIZE);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        linear_kernel<<<grid, block>>>(ctx, out_w_ptr, out_b_ptr, out_ptr, M, N, K);
    }
}

}