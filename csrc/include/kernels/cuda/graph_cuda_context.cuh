#pragma once
#include <cudnn.h>
#include <cublas_v2.h>
#include <torch/torch.h>
#include <memory>
#include <algorithm>
#include "runtime_types.hpp"

namespace fxfusion::kernels::cuda {

inline const char* cublas_get_error_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                             return "UNKNOWN_CUBLAS_STATUS";
    }
}

struct CudnnDeleter {
    void operator()(cudnnContext* h) const {
        if (h) cudnnDestroy(h);
    }
};

struct CublasDeleter {
    void operator()(cublasContext* h) const {
        if (h) cublasDestroy(h);
    }
};

using CudnnHandle  = std::unique_ptr<cudnnContext, CudnnDeleter>;
using CublasHandle = std::unique_ptr<cublasContext, CublasDeleter>;

struct GraphCudaContext : GraphContext {
    GraphCudaContext() {
        cudnnHandle_t  raw_cudnn  = nullptr;
        cublasHandle_t raw_cublas = nullptr;

        const cudnnStatus_t  cudnn_st  = cudnnCreate(&raw_cudnn);
        const cublasStatus_t cublas_st = cublasCreate(&raw_cublas);

        if (cudnn_st != CUDNN_STATUS_SUCCESS || cublas_st != CUBLAS_STATUS_SUCCESS) {
            if (raw_cudnn)  cudnnDestroy(raw_cudnn);
            if (raw_cublas) cublasDestroy(raw_cublas);

            TORCH_CHECK(cudnn_st == CUDNN_STATUS_SUCCESS,
                        "cudnnCreate failed: ", cudnnGetErrorString(cudnn_st));
            TORCH_CHECK(cublas_st == CUBLAS_STATUS_SUCCESS,
                        "cublasCreate failed: ", cublas_get_error_string(cublas_st));
        }

        cudnn_.reset(raw_cudnn);
        cublas_.reset(raw_cublas);
    }

    cudnnHandle_t  cudnn_handle()  const { return cudnn_.get(); }
    cublasHandle_t cublas_handle() const { return cublas_.get(); }

    void* workspace_ptr() const {
        return workspace_size_bytes_ > 0 ? workspace_buf_.data_ptr() : nullptr;
    }

    void update_workspace_max_size(int64_t bytes) {
        workspace_size_bytes_ = std::max(workspace_size_bytes_, bytes);
    }

    void ensure_workspace() override {
        if (workspace_size_bytes_ <= 0) return;
        if (workspace_buf_.defined() && workspace_buf_.nbytes() >= workspace_size_bytes_) {
            return;
        }
        workspace_buf_ = torch::empty(
            {workspace_size_bytes_},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8)
        );
    }

    GraphCudaContext(const GraphCudaContext&) = delete;
    GraphCudaContext& operator=(const GraphCudaContext&) = delete;

private:
    CudnnHandle   cudnn_;
    CublasHandle  cublas_;
    torch::Tensor workspace_buf_;
    int64_t       workspace_size_bytes_ = 0;
};

} // namespace fxfusion::kernels::cuda