# Graph Optimization and CUDA Execution Framework for Neural Inference

Experimental ML compiler/runtime framework for lowering PyTorch FX graphs into optimized execution pipelines using graph rewriting, operator fusion, memory planning, and custom CUDA/C++ execution backends.

## Current Status

* PyTorch FX graph tracing implemented
* Pattern-based operator fusion implemented
* Shape propagation pass implemented
* Static activation memory planner implemented
* Arena allocation with activation reuse implemented
* Alias-aware memory planning for flatten/reshape operations implemented
* FlatBuffers integration in progress
* LibTorch runtime integration in progress
* CUDA custom operator experiments in progress

## Supported Fusion Patterns (So Far)

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
 ├── Operator Registry
 ├── Execution Graph
 └── CUDA/C++ Kernels
```

## Tech Stack

* Python
* C++17
* CUDA
* PyTorch FX
* LibTorch
* FlatBuffers
* CMake
