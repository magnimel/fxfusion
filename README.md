# FXFusion — Graph Compiler and Runtime for Neural Network Inference

Compiler and runtime framework for lowering PyTorch FX graphs into optimized execution plans through graph rewriting, operator fusion, shape propagation, static memory planning, and native C++/CUDA execution backends.

FXFusion explores ahead-of-time inference execution, activation-memory reuse, precomputed operator dispatch, and integration with optimized GPU libraries. Its design draws on techniques used in systems such as TensorRT, TorchInductor, ONNX Runtime, TVM, and XLA.

## Performance

Benchmark reports:

* [Benchmark Report (CPU)](./benchmarks/README_CPU.md)
* [Benchmark Report (CUDA)](./benchmarks/README_CUDA.md)

### CPU Backend

* Up to **2.16× faster** than PyTorch eager on dispatch-heavy inference workloads
* Up to **1.67× faster** than `torch.compile` on dispatch-heavy MLP benchmarks
* Up to **1.35× faster** than PyTorch eager on dispatch-heavy GPT forward and decoding workloads
* Outperforms `torch.compile` on dispatch-heavy GPT decoding workloads
* Outperforms both PyTorch eager and `torch.compile` on balanced MLP workloads
* Performance comparable to or exceeding `torch.compile` on ResNet-18 inference
* Performance parity with PyTorch eager on ResNet-50 inference

### CUDA Backend

Preliminary results **after cuBLASLt and cuDNN integration**, measured on an NVIDIA T4:

* **Dispatch-heavy MLP:** approximately **8.3× faster** than PyTorch eager and **7.4× faster** than `torch.compile` (depth=32, width=64, batch=1)
* **Dispatch-heavy GPT forward:** approximately **8.4× faster** than PyTorch eager and **4.1× faster** than `torch.compile` (`Nx=12`, `d_model=128`, batch=1, sequence length=32)
* **Dispatch-heavy GPT decoding (full recomputation):** approximately **9.4× faster** than PyTorch eager and **4.6× faster** than `torch.compile`
* **Compute-heavy GPT forward:** approximately **3.9× faster** than PyTorch eager and **1.7× faster** than `torch.compile` (`Nx=12`, `d_model=512`, batch=2, sequence length=10)
* **Compute-heavy MLP:** near parity with `torch.compile`, where large matrix multiplications dominate execution time
* **ResNet-18:** approximately **1.2× faster** than PyTorch eager and **1.5× faster** than `torch.compile`
* **ResNet-50:** performance comparable to `torch.compile`, using cuDNN-backed convolution and pooling

These figures summarize repeated development benchmark runs, not a finalized performance evaluation. GPT decoding benchmarks use **full-sequence recomputation without a KV cache**. The linked CUDA benchmark report currently documents an earlier, pre-library-integration implementation and will be updated separately.

## Current Status

### Compiler Pipeline

* PyTorch FX graph tracing
* Multi-stage, pattern-based operator fusion
* Transformer-specific graph rewrites, including QKV packing and MHA/FeedForward fusion
* Shape and dtype propagation
* Static activation-memory planning with tensor lifetime analysis
* Best-fit arena placement with buffer reuse, free-block coalescing, and alignment
* Alias-aware memory planning and tensor binding
* FlatBuffers graph serialization

### Runtime

* Native C++ `RuntimeGraph` execution engine with precomputed `RuntimeNode` function-pointer dispatch
* Graph metadata deserialized once at engine construction; no FlatBuffers traversal in the forward pass
* `MemoryManager` with arena-backed activations, tensor registry, and alias binding
* Persistent model constants and input/output tensor management through LibTorch
* Operator-specific execution caches constructed during engine initialization
* Shared CUDA backend context for cuBLASLt and cuDNN handles and execution workspace
* CPU and CUDA execution backends
* End-to-end output validation against PyTorch on MLP, ResNet, and GPT workloads

## Transformer Support

Implemented in PyTorch as model definitions and used to exercise the FXFusion compiler and runtime:

* Transformer Encoder
* Transformer Decoder
* Encoder–Decoder Transformer
* GPT-style Decoder-Only Transformer
* Multi-Head Attention (MHA)
* FeedForward Blocks
* LayerNorm
* Static Mask Generation
* Autoregressive Decoding with full-sequence recomputation

FXFusion supports compiled execution of GPT-style Transformer workloads on CPU and CUDA. KV-cache-based incremental decoding is not yet implemented.

### CPU Backend

* Supports CNN, MLP, and Transformer inference
* Delegates selected compute-intensive operations to LibTorch and BLAS-backed implementations
* Correctness exercised on ResNet-18, ResNet-50, MLP, and GPT-style models

### CUDA Backend

* Native CUDA dispatch integrated with the C++ runtime
* **cuBLASLt** execution for Linear, Linear+ReLU, FeedForward projections, and MHA QKV/output projections
* Cached cuBLASLt operation descriptors, matrix layouts, selected algorithms, and epilogue configuration prepared at engine construction
* **cuDNN Frontend** graph-based execution for Conv2D, Conv2D+ReLU, MaxPool2D, AvgPool2D, and supported AdaptiveAvgPool2D configurations
* Cached cuDNN operation graphs and execution plans prepared during engine construction and reused for inference
* Shared workspace management for CUDA library operations
* Custom FlashAttention-style CUDA kernel for fused attention, using online softmax without materializing the full attention matrix
* Custom CUDA implementations for elementwise operations, normalization, embedding, and tensor transformations
* CUDA execution exercised end to end on MLP, ResNet-18, ResNet-50, and GPT workloads

The earlier hand-written tiled GEMM and convolution implementations have been superseded by vendor-library-backed execution for the corresponding active operators. Custom CUDA kernels remain in use where appropriate, including the fused attention computation.

## Supported Operators

### CNN

*(CPU and CUDA)*

* Conv2D
* Conv2D + ReLU
* MaxPool2D
* AvgPool2D
* AdaptiveAvgPool2D (supported configurations)

### Elementwise / Shared

*(CPU and CUDA)*

* Add
* Add + ReLU
* Mul
* ReLU

### Transformer

*(CPU and CUDA, within supported graph patterns)*

* Embedding
* Linear
* Linear + ReLU
* LayerNorm
* Add + LayerNorm
* Multi-Head Attention (MHA)
* FeedForward
* Transpose
* Narrow
* Size
* View / alias operations

Additional primitive operator coverage is being developed for executing fully unfused Transformer graphs.

## Supported Fusion Patterns

* Conv2D + BatchNorm + ReLU
* Conv2D + BatchNorm (inference-time parameter folding)
* Conv2D + ReLU
* Linear + ReLU
* Add + ReLU
* Add + LayerNorm
* QKV projection packing
* Attention
* Multi-Head Attention (MHA)
* FeedForward (Linear + ReLU + Linear)

Standalone operations such as Linear, Conv2D, ReLU, LayerNorm, and Embedding are also lowered into FXFusion-native operators.

## Architecture

```text
PyTorch Model
      │
      ▼
FX Graph Tracing
      │
      ▼
Compiler Pass Pipeline
 ├── Operator Lowering and Graph Rewriting
 ├── Multi-Stage Operator Fusion
 ├── Shape Propagation
 └── Static Memory Planning
      ├── Tensor Lifetime Analysis
      ├── Best-Fit Arena Placement and Reuse
      └── Alias Tracking
      │
      ▼
FlatBuffers Serialization
      │
      ▼
C++ Execution Engine (constructed once)
 ├── RuntimeGraph
 │    └── RuntimeNode (OpCode, TensorIds, Params, KernelFn*)
 │
 ├── MemoryManager
 │    ├── Activation Arena
 │    ├── Tensor Registry
 │    ├── Constant and Input Binding
 │    └── Alias Binding
 │
 └── Backend Context and Operator Caches
      ├── CPU / LibTorch
      └── CUDA
           ├── cuBLASLt Descriptors and Algorithms
           ├── cuDNN Frontend Graphs and Execution Plans
           ├── Shared CUDA Workspace
           └── Custom CUDA Kernels
      │
      ▼
Precomputed Native Operator Dispatch
      │
      ▼
Inference Outputs
```

## Tech Stack

* **Languages:** Python, C++17, CUDA
* **Frameworks:** PyTorch FX, LibTorch
* **Serialization:** FlatBuffers
* **Build:** CMake
* **GPU Libraries:** cuBLASLt, cuDNN, cuDNN Frontend, CUDA Runtime
* **Testing and Benchmarks:** pytest, PyTorch eager, `torch.compile`

## Testing

```bash
pytest py/tests/ -v

# or

make test
```

* `test_ops.py` — end-to-end correctness for operators and models across supported CPU/CUDA configurations
* `test_fusion.py` — fusion pattern matching and correctness
* `test_memory_plan.py` — arena size, offset alignment, reuse, alias tracking, and liveness analysis
* `test_shape_prop.py` — tensor shape and dtype propagation

## Future Work

### Compiler and Runtime

* Strict one-to-one primitive lowering and a fully unfused execution configuration
* Additional primitive operators for BatchNorm, GELU, MatMul, and attention decomposition
* Controlled fusion, dispatch, and memory-reuse configurations
* KV-cache support for incremental GPT decoding
* Multi-stream execution and CUDA Graph integration
* Extended dynamic-shape support

### CUDA Backend

* Further profiling and kernel tuning using Nsight Systems and Nsight Compute
* Broader operator and shape coverage
* Additional CUDA execution and workspace optimizations

### Models and Benchmarking

* Vision Transformers (ViT)
* Expanded CPU/CUDA benchmark coverage across model sizes and batch sizes
* End-to-end latency distributions and process-level GPU memory profiling
* Comparisons with specialized inference runtimes

## Goals

The long-term goal of FXFusion is to explore the design and implementation of machine-learning compiler and runtime systems, with an emphasis on ahead-of-time graph execution, graph optimization, memory planning, operator fusion, backend integration, and efficient CPU/GPU inference.