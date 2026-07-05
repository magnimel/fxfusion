#include "kernels.cuh"

namespace fxfusion::kernels::cuda {

__global__ void mul_tensor_scalar_kernel(const float* x, float* out, float scalar, int64_t N) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        out[idx] = x[idx] * scalar;
    }
}

__global__ void mul_tensor_tensor_kernel(const float* x, const float* y, float* out, int64_t N) {
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x)
    {
        out[idx] = x[idx] * y[idx];
    }
}

void mul(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache*) {
    const auto& x = reg[input_ids[0]];
    auto& out     = reg[output_ids[0]];

    const int64_t mul_mode = params.ints[0];
    const int64_t N = x.numel();
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    switch (mul_mode) {
        case 0: {
            const auto& y = reg[input_ids[1]];
            TORCH_CHECK(y.numel() == N, "mul: tensor-tensor shape mismatch, x.numel()=", N, " y.numel()=", y.numel());
            const float* y_ptr = y.data_ptr<float>();
            mul_tensor_tensor_kernel<<<grid, block>>>(x_ptr, y_ptr, out_ptr, N);
            break;
        }
        case 1:
        case 2:
        case 3: {
            float scalar = params.floats[0];
            mul_tensor_scalar_kernel<<<grid, block>>>(x_ptr, out_ptr, scalar, N);
            break;
        }
        default:
            throw std::runtime_error("mul: unknown mul_mode " + std::to_string(mul_mode));
    }
}

}