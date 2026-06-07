# FXFusion — Graph Optimization and Execution Framework for Neural Inference

Experimental ML compiler/runtime framework for lowering PyTorch FX graphs into optimized execution pipelines using graph rewriting, operator fusion, memory planning, and custom CUDA/C++ execution backends.

The project explores compiler and runtime techniques used in modern inference systems such as TensorRT, TorchInductor, ONNX Runtime, TVM, and XLA.

## Performance

Detailed benchmark results are available in:

- [Benchmark Report](./benchmarks/README.md)

Highlights (CPU Backend):

- Up to **2.16× faster** than PyTorch eager execution on dispatch-heavy inference workloads
- Up to **1.67× faster** than `torch.compile` on dispatch-heavy MLP benchmarks
- Outperforms both PyTorch eager execution and `torch.compile` on balanced MLP workloads
- Performance comparable to or exceeding `torch.compile` on ResNet-18 CPU inference
- Performance parity with PyTorch eager execution on ResNet-50 CPU inference

## Current Status

### Compiler Pipeline

- PyTorch FX graph tracing
- Pattern-based operator fusion pass
- Shape propagation pass
- Static activation memory planner with arena allocation, buffer reuse, and alias tracking
- FlatBuffers graph serialization

### Runtime

- RuntimeNode-based execution graph with function pointer dispatch
- FlatBuffer-free forward pass — graph deserialized once at construction, never accessed at runtime
- Arena-backed tensor registry with alias binding
- CPU and CUDA execution backends
- End-to-end correctness validated against PyTorch on ResNet18, ResNet50, and MLP workloads

### CPU Backend

- CPU backend delegates to LibTorch for compute-bound ops
- Correct on ResNet18, ResNet50, MLP

### CUDA Backend

- Backend dispatch infrastructure implemented
- Custom kernel implementations in progress (cuBLAS, cuBLASLt fused epilogues)

## Supported Fusion Patterns

- Conv2D + BatchNorm + ReLU
- Conv2D + BatchNorm
- Conv2D + ReLU
- Conv2D (standalone)
- Linear + ReLU
- Linear (standalone)
- Add + ReLU

## Architecture

```text
PyTorch Model
      │
      ▼
FX Graph Tracing
      │
      ▼
Optimization Pass Pipeline
 ├── Operator Fusion
 ├── Shape Propagation
 └── Memory Planning (arena, reuse, alias)
      │
      ▼
FlatBuffers Serialization
      │
      ▼
C++ Runtime Engine
 ├── RuntimeGraph (built once from FlatBuffers)
 │    └── RuntimeNode (op_code, input_ids, output_ids, params, KernelFn*)
 ├── MemoryManager (arena allocator, tensor registry, alias binding)
 └── Backend Dispatch (CPU / CUDA via function pointer)
      │
      ▼
CPU / CUDA Kernels
```

## Tech Stack

- Languages: Python, C++17, CUDA
- Frameworks: PyTorch FX, LibTorch
- Serialization: FlatBuffers
- Build: CMake
- GPU Libraries (in progress): cuBLAS, cuBLASLt

## Testing

```bash
pytest py/tests/ -v
#or
make test
```

- `test_ops_cpu.py` — end-to-end correctness on CPU for all ops and models
- `test_fusion.py` — fusion pattern matching and correctness
- `test_memory_plan.py` — arena sizing, offset alignment, buffer reuse, liveness
- `test_shape_prop.py` — output shapes and dtype propagation per op

## Future Work

### CUDA Backend
- cuBLASLt GEMM with fused bias+ReLU epilogue
- Custom conv2d kernels with shared-memory tiling
- Occupancy-aware launch configuration

### Models
- Vision Transformers
- Transformer encoder blocks

### Runtime
- Runtime graph caching
- Multi-stream execution

## Goals

The long-term goal of FXFusion is to explore the design and implementation of modern ML compiler and runtime systems, including graph optimization, memory planning, operator fusion, kernel generation, and custom CPU/GPU execution strategies for neural network inference.