# Node Params Layout

Each serialized `Node` stores two flat parameter arrays:

- `int_params:[int]`
- `float_params:[float]`

CPU kernels ignore CUDA grid/block entries.

`get_attr` and `output` nodes are serialized as `NoOp`. Their tensor data is embedded directly in the `Tensor` table and read by the runtime from there.

---

## NoOp

`int_params = []`
`float_params = []`

Used for graph operations with no runtime effect:

- `placeholder`
- `get_attr`
- `output`
- eval-mode `Dropout`
- `view`, `reshape`, `flatten`, `contiguous`

---

## Conv2d / Conv2dRelu

`int_params = [0:stride_h, 1:stride_w, 2:padding_h, 3:padding_w, 4:dilation_h, 5:dilation_w, 6:groups, 7:grid_x, 8:grid_y, 9:grid_z, 10:block_x, 11:block_y, 12:block_z]`
`float_params = []`

Expected inputs:

`input_ids = [x, weight, bias]`

---

## MaxPool2d

`int_params = [0:kernel_h, 1:kernel_w, 2:stride_h, 3:stride_w, 4:padding_h, 5:padding_w, 6:dilation_h, 7:dilation_w, 8:ceil_mode, 9:grid_x, 10:grid_y, 11:grid_z, 12:block_x, 13:block_y, 14:block_z]`
`float_params = []`

Expected inputs:

`input_ids = [x]`

---

## AvgPool2d

`int_params = [0:kernel_h, 1:kernel_w, 2:stride_h, 3:stride_w, 4:padding_h, 5:padding_w, 6:ceil_mode, 7:grid_x, 8:grid_y, 9:grid_z, 10:block_x, 11:block_y, 12:block_z]`
`float_params = []`

Expected inputs:

`input_ids = [x]`

---

## AdaptiveAvgPool2d

`int_params = [0:output_h, 1:output_w, 2:grid_x, 3:grid_y, 4:grid_z, 5:block_x, 6:block_y, 7:block_z]`
`float_params = []`

Expected inputs:

`input_ids = [x]`

---

## Linear / LinearRelu

`int_params = []`
`float_params = []`

Expected inputs:

`input_ids = [x, weight, bias]`

---

## Add / AddRelu

`int_params = []`
`float_params = []`

Expected inputs:

`input_ids = [a, b]`

---

## Relu

`int_params = []`
`float_params = []`

Expected inputs:

`input_ids = [x]`

---

## Mul

`int_params = [0:mul_mode]`

Modes:

`0 = tensor-tensor` → `input_ids = [a, b]`,  `float_params = []`
`1 = float scalar`  → `input_ids = [x]`,     `float_params = [scalar]`
`2 = int scalar`    → `input_ids = [x]`,     `float_params = [scalar]`
`3 = bool scalar`   → `input_ids = [x]`,     `float_params = [1.0 or 0.0]`

---

## Transpose

`int_params = [0:dim0, 1:dim1]`
`float_params = []`

Expected inputs:

`input_ids = [x]`

---

## Size

`int_params = [0:dim]`
`float_params = []`

Expected inputs:

`input_ids = [x]`

Output is a scalar `Int32` tensor holding `x.size(dim)`.

Example — `x.size(1)`:

`int_params = [1]`

---

## Narrow

`int_params = [0:dim, 1:start]`
`float_params = []`

Expected inputs:

`input_ids = [x, length]`

`length` is the scalar `Int32` tensor produced by a `Size` node.

The kernel computes:

`x.narrow(dim, start, length_tensor.item())`

The output is a runtime view into `x` — no copy is made.

Example — `x.narrow(dim=1, start=0, length=<runtime>)`:

`int_params = [1, 0]`

---

## Embedding

`int_params = []`
`float_params = []`

Expected inputs:

`input_ids = [indices, weight]`

---

## LayerNorm

`int_params = [0:normalized_rank, 1..rank:normalized_shape_dims]`
`float_params = [0:eps]`

Expected inputs:

`input_ids = [x, weight, bias]`

Example — `normalized_shape = [512]`:

`int_params = [1, 512]`
`float_params = [1e-05]`

---

## AddLayerNorm

`int_params = [0:normalized_rank, 1..rank:normalized_shape_dims]`
`float_params = [0:eps]`

Expected inputs:

`input_ids = [a, b, weight, bias]`

The kernel computes:

`layer_norm(a + b, normalized_shape, weight, bias, eps)`

Example — `normalized_shape = [512]`:

`int_params = [1, 512]`
`float_params = [1e-05]`

---

## MHA

`int_params = [0:num_heads, 1:head_dim, 2:d_model, 3:qkv_dim]`
`float_params = [0:scale_divisor]`

Expected inputs:

`input_ids = [x, mask, qkv_weight, qkv_bias, out_weight, out_bias]`

Mask convention: `True = keep`, `False = blocked`.

The kernel applies:

`scores.masked_fill(mask == 0, -inf)`

Mask is always required. Pass an all-ones boolean mask if no masking is needed.

Common mask shapes:

`[batch, 1, seq_len, seq_len]` — causal / decoder attention
`[batch, 1, 1, seq_len]`       — padding / encoder attention

Example — 8-head attention, `d_model = 512`:

`int_params = [8, 64, 512, 1536]`
`float_params = [8.0]`

---

## FeedForward

`int_params = []`
`float_params = []`

Expected inputs:

`input_ids = [x, w1, b1, w2, b2]`

The kernel computes:

`linear(relu(linear(x, w1, b1)), w2, b2)`