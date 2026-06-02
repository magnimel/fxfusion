# Graph Optimization and CUDA Execution Framework for Neural Inference

Experimental ML compiler/runtime framework for lowering PyTorch FX graphs into optimized execution pipelines using graph rewriting, operator fusion, memory planning, and custom CUDA/C++ execution backends.

The project explores compiler and runtime techniques used in modern inference systems such as TensorRT, TorchInductor, ONNX Runtime, TVM, and XLA.

## Current Status

### Compiler Pipeline

* PyTorch FX graph tracing implemented
* Pattern-based operator fusion implemented
* Shape propagation pass implemented
* Static activation memory planner implemented
* Arena allocation with activation reuse implemented
* Alias-aware memory planning implemented
* FlatBuffers graph serialization implemented

### Runtime

* FlatBuffers graph loading implemented
* LibTorch runtime integration implemented
* Arena allocator implemented
* Tensor registry implemented
* Operator registry implemented
* Alias binding support implemented
* CPU execution backend implemented
* CUDA execution backend implemented
* End-to-end execution validated against PyTorch

### CPU Backend

* BLAS-backed Linear kernel implemented
* BLAS-backed LinearReLU kernel implemented
* Vectorized bias/ReLU epilogue implemented
* End-to-end CPU execution validated against PyTorch
* Additional CPU operators in progress

### CUDA Backend

* CUDA kernel implementations in progress

## Supported Fusion Patterns

* Conv2D + BatchNorm
* Conv2D + BatchNorm + ReLU
* Linear + ReLU
* Add + ReLU

## Architecture

```text
PyTorch Model
      │
      ▼
Torch FX Graph Tracing
      │
      ▼
Optimization Pass Pipeline
 ├── Operator Fusion
 ├── Shape Propagation
 ├── Memory Planning
 └── Graph Lowering
      │
      ▼
FlatBuffers Serialization
      │
      ▼
C++ Runtime Engine
 ├── Arena Allocator
 ├── Tensor Registry
 ├── Operator Registry
 └── Backend Dispatch
      │
      ▼
CPU / CUDA Kernels
```

## Tech Stack

* Languages: Python, C++17, CUDA
* Frameworks: PyTorch FX, LibTorch
* Serialization: FlatBuffers
* Build: CMake
* Performance: BLAS
* GPU Libraries (Planned): cuBLAS, cuBLASLt

## Future Work

### Runtime

* RuntimeNode execution graph
* Function-pointer dispatch
* Runtime graph caching
* FlatBuffer-free execution path

### CPU Backend

* Additional operator implementations
* BLAS-backed operator expansion

### CUDA Backend

* Custom kernel implementations
* Shared-memory tiling
* Occupancy-aware scheduling
* cuBLAS integration
* Kernel autotuning

### Models

* ResNet18
* ResNet50
* Vision Transformers
* Transformer encoder blocks

## Goals

The long-term goal of FXFusion is to explore the design and implementation of modern ML compiler and runtime systems, including graph optimization, memory planning, operator fusion, kernel generation, and custom CPU/GPU execution strategies for neural network inference.