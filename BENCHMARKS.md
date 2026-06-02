# FXFusion Benchmarks

## TinyMLP Validation

Model:

```python
Linear(4096, 4096)
ReLU()
Linear(4096, 4096)
ReLU()
Linear(4096, 4096)
ReLU()
```

### Lowered FX Graph

```text
placeholder    x

get_attr       layers_0_fused_weight
get_attr       layers_0_fused_bias

fused_linear_relu

get_attr       layers_2_fused_weight
get_attr       layers_2_fused_bias

fused_linear_relu

get_attr       layers_4_fused_weight
get_attr       layers_4_fused_bias

fused_linear_relu

output
```

### Memory Plan

```text
ARENA SIZE: 2097152

Node Name                           | Kind         | Size (B)   | Offset
---------------------------------------------------------------------------
x                                   | input        | 1048576    | N/A
layers_0_fused_weight               | const        | 67108864   | N/A
layers_0_fused_bias                 | const        | 16384      | N/A
fused_linear_relu                   | activation   | 1048576    | 0
layers_2_fused_weight               | const        | 67108864   | N/A
layers_2_fused_bias                 | const        | 16384      | N/A
fused_linear_relu_1                 | activation   | 1048576    | 1048576
layers_4_fused_weight               | const        | 67108864   | N/A
layers_4_fused_bias                 | const        | 16384      | N/A
fused_linear_relu_2                 | activation   | 1048576    | 0
output                              | output       | 1048576    | Alias
```

### Correctness

```text
[True] Success
```

Output matches PyTorch within tolerance:

```python
torch.allclose(
    fxfusion_output,
    pytorch_output,
    rtol=1e-4,
    atol=1e-5
)
```

### Performance

```text
PyTorch      : 18.7235 ms
FXFusion     : 18.8217 ms
torch.compile: 18.6301 ms
```

### Relative Performance

```text
FXFusion vs PyTorch      : 0.99x
FXFusion vs torch.compile: 0.99x
```

### CPU Backend Configuration

- BLAS-backed GEMM
- Vectorized bias addition
- Vectorized ReLU epilogue
- Arena-based activation storage
- Activation memory reuse

### Compiler Vectorization Report

```text
remark: vectorized loop
vectorization width: 4
interleaved count: 4
```

## Notes

Current benchmark demonstrates:

- End-to-end graph lowering
- FlatBuffers serialization
- Runtime graph loading
- Arena allocation
- Tensor registry execution
- BLAS-backed fused LinearReLU kernels
- Numerical equivalence with PyTorch

The CPU backend currently performs within approximately 1% of both PyTorch eager execution and torch.compile on this workload.