#include "kernels.cuh"
#include "internal/elementwise_kernels.cuh"

namespace fxfusion::kernels::cuda {

void mul(TensorRegistry& reg, const TensorIds& input_ids, const TensorIds& output_ids, const Params& params, const Cache*) {
    const auto& x = reg[input_ids[0]];
    auto& out     = reg[output_ids[0]];
    
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    const int64_t mul_mode = params.ints[0];
    
    int64_t N = x.numel();

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    switch (mul_mode) {
        case 0: {
            const auto& y = reg[input_ids[1]];
            TORCH_CHECK(y.numel() == N, "mul[cuda]: tensor-tensor shape mismatch, x.numel()=", N, " y.numel()=", y.numel());
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
            throw std::runtime_error("mul[cuda]: unknown mul_mode " + std::to_string(mul_mode));
    }
}

} // namespace fxfusion::kernels::cuda