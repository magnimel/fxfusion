# FXFusion — Graph Compiler and Runtime for Neural Network Inference

Compiler and runtime framework for lowering PyTorch FX graphs into optimized execution plans through graph rewriting, operator fusion, shape propagation, memory planning, and custom C++/CUDA execution backends.

The project explores compiler and runtime techniques used in modern inference systems such as TensorRT, TorchInductor, ONNX Runtime, TVM, and XLA.

## Performance

Detailed benchmark results are available in:

* [Benchmark Report](./benchmarks/README.md)

Highlights (CPU Backend):

* Up to **2.16× faster** than PyTorch eager execution on dispatch-heavy inference workloads
* Up to **1.67× faster** than `torch.compile` on dispatch-heavy MLP benchmarks
* Up to **1.35× faster** than PyTorch eager execution on dispatch-heavy GPT forward and decoding workloads
* Outperforms `torch.compile` on dispatch-heavy GPT decoding workloads
* Outperforms both PyTorch eager execution and `torch.compile` on balanced MLP workloads
* Performance comparable to or exceeding `torch.compile` on ResNet-18 CPU inference
* Performance parity with PyTorch eager execution on ResNet-50 CPU inference

## Current Status

### Compiler Pipeline

* PyTorch FX graph tracing
* Multi-stage pattern-based operator fusion pass
* Transformer-specific graph rewrites and operator packing
* Shape propagation pass
* Static activation memory planner with arena allocation, buffer reuse, and alias tracking
* FlatBuffers graph serialization

### Runtime

* RuntimeGraph execution engine with RuntimeNode function-pointer dispatch
* FlatBuffer-free forward pass — graph deserialized once at construction, never accessed at runtime
* Arena-backed tensor registry with alias binding
* Transformer runtime support including MHA, FeedForward, LayerNorm, Embedding, and autoregressive GPT decoding
* CPU and CUDA execution backends
* End-to-end correctness validated against PyTorch on ResNet18, ResNet50, MLP, and Transformer workloads

## Transformer Support

Implemented from scratch in PyTorch:

* Transformer Encoder
* Transformer Decoder
* Encoder–Decoder Transformer
* GPT-style Decoder-Only Transformer
* Multi-Head Attention (MHA)
* FeedForward Blocks
* LayerNorm
* Static Mask Generation
* Autoregressive Decoding

Validated through FXFusion execution against PyTorch eager execution.

### CPU Backend

* CPU backend supports CNN, MLP, and Transformer execution
* CPU backend delegates compute-bound operators to LibTorch
* Correctness validated on ResNet18, ResNet50, MLP, and GPT-style Transformer models

### CUDA Backend

* Backend dispatch infrastructure implemented
* Custom kernel implementations in progress
* cuBLAS and cuBLASLt integration in progress

## Supported Operators

### CNN

* Conv2D
* Conv2D + ReLU
* MaxPool2D
* AvgPool2D
* AdaptiveAvgPool2D
* Add
* Add + ReLU

### Transformer

* Embedding
* Linear
* Linear + ReLU
* LayerNorm
* Add + LayerNorm
* Multi-Head Attention (MHA)
* FeedForward
* Mul
* Transpose
* Narrow
* Size

## Supported Fusion Patterns

* Conv2D + BatchNorm + ReLU
* Conv2D + BatchNorm
* Conv2D + ReLU
* Linear + ReLU
* Add + ReLU
* LayerNorm
* Add + LayerNorm
* FeedForward (Linear + ReLU + Linear)
* QKV Linear Packing

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
 ├── Transformer Graph Rewrites
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
 │
 ├── MemoryManager
 │    ├── Arena Allocation
 │    ├── Tensor Registry
 │    └── Alias Binding
 │
 └── Backend Dispatch (CPU / CUDA via function pointer)
      │
      ▼
CPU / CUDA Kernels
```

## Tech Stack

* Languages: Python, C++17, CUDA
* Frameworks: PyTorch FX, LibTorch
* Serialization: FlatBuffers
* Build: CMake
* GPU Libraries (in progress): cuBLAS, cuBLASLt

## Testing

```bash
pytest py/tests/ -v

# or

make test
```

* `test_ops_cpu.py` — end-to-end correctness on CPU for all operators and models
* `test_fusion.py` — fusion pattern matching and correctness
* `test_memory_plan.py` — arena sizing, offset alignment, buffer reuse, alias tracking, and liveness analysis
* `test_shape_prop.py` — output shape and dtype propagation

## Future Work

### CUDA Backend

* cuBLASLt GEMM with fused bias + ReLU epilogues
* Custom Conv2D kernels with shared-memory tiling
* Occupancy-aware launch configuration
* Fused Transformer kernels

### Models

* Vision Transformers (ViT)
* BERT-style encoder architectures
* LLaMA-style decoder architectures

### Runtime

* KV-cache support for autoregressive decoding
* Runtime graph caching
* Multi-stream execution
* CUDA Graph integration

## Goals

The long-term goal of FXFusion is to explore the design and implementation of modern ML compiler and runtime systems, including graph optimization, memory planning, operator fusion, graph lowering, kernel generation, and custom CPU/GPU execution strategies for neural network inference.
