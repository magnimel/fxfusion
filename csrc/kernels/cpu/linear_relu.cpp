#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include "kernels.hpp"

namespace fxfusion::kernels::cpu {

void linear_relu(
    const std::vector<torch::Tensor>& registry, 
    const fxfusion::Node* node
) {
    const auto& x = registry[node->inputs()->Get(0)];
    const auto& w = registry[node->inputs()->Get(1)];
    const auto& b = registry[node->inputs()->Get(2)];
    auto& out     = registry[node->outputs()->Get(0)];

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = w.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr     = out.data_ptr<float>();


    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w.size(0);


    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        M, N, K,
        1.0f,
        x_ptr, K,
        w_ptr, K,
        0.0f,
        out_ptr, N
    );

    constexpr int TILE = 16;

    for (int m0 = 0; m0 < M; m0 += TILE) {
        for (int n0 = 0; n0 < N; n0 += TILE) {

            const int m_end = std::min(m0 + TILE, M);
            const int n_end = std::min(n0 + TILE, N);

            for (int m = m0; m < m_end; ++m) {
                const int row_offset = m * N;

                for (int n = n0; n < n_end; ++n) {
                    const int idx = row_offset + n;

                    out_ptr[idx] += b_ptr[n];
                    out_ptr[idx] = std::max(out_ptr[idx], 0.0f);
                }
            }
        }
    }

    
}

}