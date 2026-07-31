#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

// Input:  x      {batch, C_in, H_in, W_in}
//         w      {C_out, C_in / groups, ksize_h, ksize_w}
//         b      {C_out}
// Output: out    {batch, C_out, H_out, W_out}
__global__ void conv2d_relu_kernel(
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
        out[out_idx] = fmaxf(0.0f, res + b[c_out]);
    }
}

void conv2d_relu (TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache* cache_base) {
    const auto& x   = reg[input_ids[0]];
    const auto& w   = reg[input_ids[1]];
    const auto& b   = reg[input_ids[2]];
    auto& out       = reg[output_ids[0]];

    const std::vector<int64_t> stride   = {params.ints[0], params.ints[1]};
    const std::vector<int64_t> padding  = {params.ints[2], params.ints[3]};
    const std::vector<int64_t> dilation = {params.ints[4], params.ints[5]};
    const int64_t groups                =  params.ints[6];

    const float* x_ptr   = x.data_ptr<float>();
    const float* w_ptr   = w.data_ptr<float>();
    const float* b_ptr   = b.data_ptr<float>();
    float* out_ptr       = out.data_ptr<float>();
    
    int64_t N     = x.size(0);
    int64_t C_in  = x.size(1);
    int64_t H_in  = x.size(2);
    int64_t W_in  = x.size(3);

    int64_t C_out = out.size(1);
    int64_t H_out = out.size(2);
    int64_t W_out = out.size(3);

    int64_t ksize_h = w.size(2);
    int64_t ksize_w = w.size(3);

    int64_t xtsize_h = (CONV2D_TILE_DIM - 1) * stride[0] + (ksize_h - 1) * dilation[0] + 1;
    int64_t xtsize_w = (CONV2D_TILE_DIM - 1) * stride[1] + (ksize_w - 1) * dilation[1] + 1;

    size_t shared_bytes = (ksize_h * ksize_w + xtsize_h * xtsize_w ) * sizeof(float);

    dim3 block(CONV2D_TILE_DIM, CONV2D_TILE_DIM);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C_out);

    conv2d_relu_kernel<<<grid, block, shared_bytes>>>(
        x_ptr, w_ptr, b_ptr, out_ptr,
        ksize_h, ksize_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        groups,
        C_in, H_in, W_in,
        C_out, H_out, W_out
    );
      
}

} 