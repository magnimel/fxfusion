#pragma once
#include <cudnn.h>
#include <cublasLt.h>
#include <torch/torch.h>
#include <memory>
#include <algorithm>
#include "runtime_types.hpp"

namespace fxfusion::kernels::cuda {

#define CUBLASLT_CHECK(status)                                              \
    do {                                                                    \
        cublasStatus_t _status = (status);                                  \
        TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS,                       \
                    "cuBLASLt error: ", cublasLtGetStatusString(_status));  \
    } while (0)

struct CudnnDeleter {
    void operator()(cudnnContext* h) const {
        if (h) cudnnDestroy(h);
    }
};

struct CublasLtDeleter {
    void operator()(cublasLtContext* h) const {
        if (h) cublasLtDestroy(h);
    }
};

using CudnnHandle    = std::unique_ptr<cudnnContext, CudnnDeleter>;
using CublasLtHandle = std::unique_ptr<cublasLtContext, CublasLtDeleter>;

struct GraphCudaContext : GraphContext {
    GraphCudaContext() {
        cudnnHandle_t    raw_cudnn    = nullptr;
        cublasLtHandle_t raw_cublasLt = nullptr;

        const cudnnStatus_t  cudnn_st    = cudnnCreate(&raw_cudnn);
        const cublasStatus_t cublasLt_st = cublasLtCreate(&raw_cublasLt);

        if (cudnn_st != CUDNN_STATUS_SUCCESS || cublasLt_st != CUBLAS_STATUS_SUCCESS) {
            if (raw_cudnn)    cudnnDestroy(raw_cudnn);
            if (raw_cublasLt) cublasLtDestroy(raw_cublasLt);

            TORCH_CHECK(cudnn_st == CUDNN_STATUS_SUCCESS,
                        "cudnnCreate failed: ", cudnnGetErrorString(cudnn_st));
            TORCH_CHECK(cublasLt_st == CUBLAS_STATUS_SUCCESS,
                        "cublasLtCreate failed: ", cublasLtGetStatusString(cublasLt_st));
        }

        cudnn_.reset(raw_cudnn);
        cublasLt_.reset(raw_cublasLt);
    }

    cudnnHandle_t    cudnn_handle()    const { return cudnn_.get(); }
    cublasLtHandle_t cublasLt_handle() const { return cublasLt_.get(); }

    void* workspace_ptr() const {
        return workspace_size_bytes_ > 0 ? workspace_buf_.data_ptr() : nullptr;
    }

    size_t workspace_size() const { return workspace_size_bytes_; }

    template <typename T>
    void update_workspace_max_size(T bytes) {
        if (bytes > 0) {
            workspace_size_bytes_ = std::max(workspace_size_bytes_, static_cast<size_t>(bytes));
        }
    }   

    void ensure_workspace() override {
        if (workspace_size_bytes_ == 0) return;
        if (workspace_buf_.defined() && workspace_buf_.nbytes() >= workspace_size_bytes_) {
            return;
        }
        workspace_buf_ = torch::empty(
            {static_cast<int64_t>(workspace_size_bytes_)},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8));
    }

    GraphCudaContext(const GraphCudaContext&) = delete;
    GraphCudaContext& operator=(const GraphCudaContext&) = delete;

private:
    CudnnHandle    cudnn_;
    CublasLtHandle cublasLt_;
    torch::Tensor  workspace_buf_;
    size_t        workspace_size_bytes_ = 0;
};

} // namespace fxfusion::kernels::cuda