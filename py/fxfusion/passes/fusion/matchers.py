import math
import operator

import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, cast

from fxfusion.passes.fusion.types import (
    FusionContext,
    FusionSpec,
    PatternItem
)

from fxfusion.passes.fusion.symbols import Fusion
# ============================================================
# UTILITY HELPERS
# ============================================================

def _get_node_chain(tail_node: fx.Node, depth: int) -> Optional[List[fx.Node]]:
    chain = [tail_node]
    curr = tail_node

    for _ in range(depth - 1):
        if not curr.args or not isinstance(curr.args[0], fx.Node):
            return None

        curr = curr.args[0]
        chain.append(curr)

    return list(reversed(chain))


def _get_node_chain_forward(head_node: fx.Node, depth: int) -> Optional[List[fx.Node]]:
    chain = [head_node]
    curr = head_node

    for _ in range(depth - 1):
        users = list(curr.users)
        if len(users) != 1 or not isinstance(users[0], fx.Node):
            return None

        curr = users[0]
        chain.append(curr)

    return list(chain)


def _match_chain_forward(
    ctx: FusionContext,
    head_node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[List[fx.Node]]:
    chain = _get_node_chain_forward(head_node, len(pattern))
    if chain is None:
        return None

    for expected_ops, current_node in zip(pattern, chain):
        if not any(_match_node(ctx, op, current_node) for op in expected_ops):
            return None

    return chain

def _match_node(ctx: FusionContext, expected_op, current_node: fx.Node) -> bool:
    if isinstance(expected_op, type):
        if current_node.op != "call_module":
            return False

        if not isinstance(current_node.target, str):
            return False

        target_mod = ctx.modules.get(current_node.target)
        return isinstance(target_mod, expected_op)

    if isinstance(expected_op, str):
        return current_node.op == "call_method" and current_node.target == expected_op
    
    if current_node.op != "call_function":
        return False

    return current_node.target == expected_op

def _match_chain_pattern(
    ctx: FusionContext,
    tail_node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[List[fx.Node]]:
    chain = _get_node_chain(tail_node, len(pattern))

    if chain is None:
        return None

    for expected_ops, current_node in zip(pattern, chain):
        if not any(_match_node(ctx, op, current_node) for op in expected_ops):
            return None

    return chain

def _has_external_users(chain: List[fx.Node]) -> bool:
    chain_set = set(chain)
    for node in chain[:-1]:
        external_users = [user for user in node.users if user not in chain_set]
        if external_users:
            return True
    return False

def _conv_extra(conv_mod: nn.Conv2d) -> Dict[str, Any]:
    return {
        "stride": conv_mod.stride,
        "padding": conv_mod.padding,
        "dilation": conv_mod.dilation,
        "groups": conv_mod.groups,
    }

def _fuse_conv_bn_weights(
    conv_w: torch.Tensor,
    conv_b: Optional[torch.Tensor],
    bn_rm: torch.Tensor,
    bn_rv: torch.Tensor,
    bn_eps: float,
    bn_w: Optional[torch.Tensor],
    bn_b: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolves fused scaling mathematics uniformly across runtime hardware schemas."""
    target_dtype = conv_w.dtype
    target_device = conv_w.device

    bn_rm = bn_rm.to(device=target_device, dtype=target_dtype)
    bn_rv = bn_rv.to(device=target_device, dtype=target_dtype)

    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm, dtype=target_dtype, device=target_device)
    else:
        conv_b = conv_b.to(device=target_device, dtype=target_dtype)

    if bn_w is None:
        bn_w = torch.ones_like(bn_rm, dtype=target_dtype, device=target_device)
    else:
        bn_w = bn_w.to(device=target_device, dtype=target_dtype)

    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm, dtype=target_dtype, device=target_device)
    else:
        bn_b = bn_b.to(device=target_device, dtype=target_dtype)

    scale = bn_w / torch.sqrt(bn_rv + bn_eps)
    fused_weight = conv_w * scale.view(-1, 1, 1, 1)
    fused_bias = (conv_b - bn_rm) * scale + bn_b

    return (
        fused_weight.to(dtype=target_dtype, device=target_device).detach(), 
        fused_bias.to(dtype=target_dtype, device=target_device).detach()
    )

def _fetch_attr(ctx: FusionContext, get_attr_node: fx.Node) -> torch.Tensor:
    attr = ctx.fx_model
    for atom in str(get_attr_node.target).split('.'):
        if not hasattr(attr, atom):
            raise RuntimeError(f"Nonexistent attr: {get_attr_node.target}")
        attr = getattr(attr, atom)
    return attr

# ============================================================
# ATOM MATCHERS (Straight-Line Chains)
# ============================================================

def match_conv_bn_relu(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)
    if chain is None or _has_external_users(chain): return None

    conv_node, bn_node, relu_node = chain
    conv_mod = cast(nn.Conv2d, ctx.modules[str(conv_node.target)])
    bn_mod = cast(nn.BatchNorm2d, ctx.modules[str(bn_node.target)])

    weight, bias = _fuse_conv_bn_weights(
        conv_mod.weight, conv_mod.bias, bn_mod.running_mean, bn_mod.running_var,
        bn_mod.eps, bn_mod.weight, bn_mod.bias
    )

    return FusionSpec(
        tail_node=relu_node, primary_node=conv_node, input_args=(conv_node.args[0],),
        weight=weight, bias=bias, nodes_to_erase=[relu_node, bn_node, conv_node],
        extra=_conv_extra(conv_mod),
    )

def match_linear_relu(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)
    if chain is None or _has_external_users(chain): return None

    linear_node, relu_node = chain
    linear_mod = cast(nn.Linear, ctx.modules[str(linear_node.target)])

    weight = linear_mod.weight.detach()
    bias = linear_mod.bias.detach() if linear_mod.bias is not None else torch.zeros(
        linear_mod.out_features, device=weight.device, dtype=weight.dtype
    )

    return FusionSpec(
        tail_node=relu_node, primary_node=linear_node, input_args=(linear_node.args[0],),
        weight=weight, bias=bias, nodes_to_erase=[relu_node, linear_node],
    )

def match_conv_bn(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)
    if chain is None or _has_external_users(chain): return None

    conv_node, bn_node = chain
    conv_mod = cast(nn.Conv2d, ctx.modules[str(conv_node.target)])
    bn_mod = cast(nn.BatchNorm2d, ctx.modules[str(bn_node.target)])

    weight, bias = _fuse_conv_bn_weights(
        conv_mod.weight, conv_mod.bias, bn_mod.running_mean, bn_mod.running_var,
        bn_mod.eps, bn_mod.weight, bn_mod.bias
    )

    return FusionSpec(
        tail_node=bn_node, primary_node=conv_node, input_args=(conv_node.args[0],),
        weight=weight, bias=bias, nodes_to_erase=[bn_node, conv_node],
        extra=_conv_extra(conv_mod),
    )

def match_add_relu(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)
    if chain is None or _has_external_users(chain): return None

    add_node, relu_node = chain
    return FusionSpec(
        tail_node=relu_node, primary_node=add_node, input_args=tuple(add_node.args,),
        nodes_to_erase=[relu_node, add_node],
    )

def match_conv_relu(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)
    if chain is None or _has_external_users(chain): return None

    conv_node, relu_node = chain
    conv_mod = cast(nn.Conv2d, ctx.modules[str(conv_node.target)])

    weight = conv_mod.weight.detach()
    bias = conv_mod.bias.detach() if conv_mod.bias is not None else torch.zeros(
        conv_mod.out_channels, device=weight.device, dtype=weight.dtype
    )

    return FusionSpec(
        tail_node=relu_node, primary_node=conv_node, input_args=(conv_node.args[0],),
        weight=weight, bias=bias, nodes_to_erase=[relu_node, conv_node],
        extra=_conv_extra(conv_mod),
    )

def match_conv(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)
    if chain is None or _has_external_users(chain): return None

    conv_node, = chain
    conv_mod = cast(nn.Conv2d, ctx.modules[str(conv_node.target)])

    weight = conv_mod.weight.detach()
    bias = conv_mod.bias.detach() if conv_mod.bias is not None else torch.zeros(
        conv_mod.out_channels, device=weight.device, dtype=weight.dtype
    )

    return FusionSpec(
        tail_node=conv_node, primary_node=conv_node, input_args=(conv_node.args[0],),
        weight=weight, bias=bias, nodes_to_erase=[conv_node], extra=_conv_extra(conv_mod),
    )

def match_linear(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)
    if chain is None or _has_external_users(chain): return None

    linear_node, = chain
    linear_mod = cast(nn.Linear, ctx.modules[str(linear_node.target)])

    weight = linear_mod.weight.detach()
    bias = linear_mod.bias.detach() if linear_mod.bias is not None else torch.zeros(
        linear_mod.out_features, device=weight.device, dtype=weight.dtype
    )

    return FusionSpec(
        tail_node=linear_node, primary_node=linear_node, input_args=(linear_node.args[0],),
        weight=weight, bias=bias, nodes_to_erase=[linear_node],
    )

def match_add_layernorm(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)
    if chain is None or _has_external_users(chain): return None

    add_node, norm_node = chain
    norm_mod = cast(nn.LayerNorm, ctx.modules[str(norm_node.target)])

    normalized_shape = tuple(norm_mod.normalized_shape)

    weight = norm_mod.weight.detach() if norm_mod.weight is not None else torch.ones(
        normalized_shape,
        dtype=torch.float32,
    )

    bias = norm_mod.bias.detach() if norm_mod.bias is not None else torch.zeros(
        normalized_shape,
        dtype=weight.dtype,
        device=weight.device,
    )

    extra = {
        "normalized_shape": list(normalized_shape),
        "eps": norm_mod.eps,
    }

    return FusionSpec(
        tail_node=norm_node, primary_node=add_node, input_args=(add_node.args),
        weight=weight, bias=bias, nodes_to_erase=[norm_node, add_node], extra=extra,
    )

def match_embedding(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None or _has_external_users(chain):
        return None

    embedding_node, = chain
    embedding_mod = cast(nn.Embedding, ctx.modules[str(embedding_node.target)])

    if embedding_mod.max_norm is not None:
        return None

    weight = embedding_mod.weight.detach()

    return FusionSpec(
        tail_node=embedding_node,
        primary_node=embedding_node,
        input_args=(embedding_node.args[0],),
        weight=weight,
        bias=None,
        nodes_to_erase=[embedding_node],
        extra=None,
    )

# ============================================================
# MOLECULE MATCHERS (DAG Matchers)
# ============================================================

def match_qkv_linear(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    
    _QKV_TAIL_PATTERN: Tuple[PatternItem, ...] = (
        (Fusion.fused_linear,),
        ("view",),
        ("transpose",),
    )
    
    if node.op != "call_function" or node.target != Fusion.fused_linear:
        return None
    
    input_node = node.args[0]
    if not isinstance(input_node, fx.Node):
        return None
    
    linear_siblings = [
        u for u in input_node.users
        if u.op == "call_function" and u.target == Fusion.fused_linear
    ]

    if len(linear_siblings) != 3:
        return None
    
    linear_siblings = sorted(linear_siblings, key=lambda n: n.name)
    if node is not linear_siblings[0]:
        return None
    
    q_node, k_node, v_node = linear_siblings
    
    q_chain = _match_chain_forward(ctx, q_node, _QKV_TAIL_PATTERN)
    k_chain = _match_chain_forward(ctx, k_node, _QKV_TAIL_PATTERN)
    v_chain = _match_chain_forward(ctx, v_node, _QKV_TAIL_PATTERN)
    
    if q_chain is None or k_chain is None or v_chain is None:
        return None
    
    packed_weight = torch.cat([
        _fetch_attr(ctx, q_node.args[1]),
        _fetch_attr(ctx, k_node.args[1]),
        _fetch_attr(ctx, v_node.args[1]),
    ], dim=0)
    
    packed_bias = torch.cat([
        _fetch_attr(ctx, q_node.args[2]),
        _fetch_attr(ctx, k_node.args[2]),
        _fetch_attr(ctx, v_node.args[2]),
    ], dim=0)

    q_view = q_chain[1]

    num_heads = q_view.args[3]
    head_dim = q_view.args[4]
    d_model = packed_weight.shape[1]

    return FusionSpec(
        tail_node=q_chain[2],
        primary_node=q_node,
        input_args=(input_node,),
        weight=packed_weight,
        bias=packed_bias,
        nodes_to_erase=[
            q_chain[2], k_chain[2], v_chain[2],
            q_chain[1], k_chain[1], v_chain[1],
            q_chain[0], k_chain[0], v_chain[0],
            q_node.args[1], q_node.args[2],
            k_node.args[1], k_node.args[2],
            v_node.args[1], v_node.args[2],
        ],
        replace_nodes=[q_chain[2], k_chain[2], v_chain[2]],
        extra={
            "d_model": d_model,
            "qkv_dim": packed_weight.shape[0],
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
    )
    

def match_attention(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:

    if node.op != "call_function" or node.target != Fusion.fused_qkv_linear:
        return None

    qkv_node = node

    qkv_input = qkv_node.args[0]
    qkv_weight_node = qkv_node.args[1]
    qkv_bias_node = qkv_node.args[2]
    qkv_extra = qkv_node.args[3]

    packed_weight = _fetch_attr(ctx, qkv_weight_node)
    packed_bias = _fetch_attr(ctx, qkv_bias_node)

    # Find K^T branch: qkv -> transpose
    kt_users = [
        u for u in qkv_node.users
        if u.op == "call_method" and u.target == "transpose"
    ]

    if len(kt_users) != 1:
        return None

    kt_node = kt_users[0]

    # Find QK matmul: matmul(qkv, kt)
    qk_users = [
        u for u in kt_node.users
        if u.op == "call_function" and u.target == torch.matmul
    ]

    if len(qk_users) != 1:
        return None

    qk_matmul_node = qk_users[0]

    if qk_matmul_node.args[0] is not qkv_node:
        return None

    if qk_matmul_node.args[1] is not kt_node:
        return None

    _ATTENTION_TAIL_PATTERN: Tuple[PatternItem, ...] = (
        (torch.matmul,),
        (operator.truediv,),
        ("masked_fill",),
        (F.softmax,),
        (nn.Dropout,),
        (torch.matmul,),
        ("transpose",),
        ("contiguous",),
        ("view",),
    )

    attn_chain = _match_chain_forward(ctx, qk_matmul_node, _ATTENTION_TAIL_PATTERN)

    if attn_chain is None:
        return None

    (
        qk_matmul_node,
        scale_node,
        masked_fill_node,
        softmax_node,
        dropout_node,
        av_matmul_node,
        out_transpose_node,
        contiguous_node,
        view_node,
    ) = attn_chain

    scale_divisor = scale_node.args[1]

    # Required mask path: eq(mask, 0) -> masked_fill(scores, eq, -inf)
    eq_node = masked_fill_node.args[1]

    if not isinstance(eq_node, fx.Node):
        return None

    if eq_node.op != "call_function" or eq_node.target != operator.eq:
        return None

    mask_input = eq_node.args[0]

    # Check AV matmul: matmul(dropout, qkv)
    if av_matmul_node.args[0] is not dropout_node:
        return None

    if av_matmul_node.args[1] is not qkv_node:
        return None

    return FusionSpec(
        tail_node=view_node,
        primary_node=qkv_node,
        input_args=(qkv_input, mask_input),
        weight=packed_weight,
        bias=packed_bias,
        nodes_to_erase=[
            view_node,
            contiguous_node,
            out_transpose_node,
            av_matmul_node,
            dropout_node,
            softmax_node,
            masked_fill_node,
            eq_node,
            scale_node,
            qk_matmul_node,
            kt_node,
            qkv_node,
            qkv_weight_node,
            qkv_bias_node,
        ],
        replace_nodes=[view_node],
        extra={
            "d_model": qkv_extra["d_model"],
            "qkv_dim": qkv_extra["qkv_dim"],
            "num_heads": qkv_extra["num_heads"],
            "head_dim": qkv_extra["head_dim"],
            "scale_divisor": scale_divisor,
        },
    )
    
    
def match_residual_add(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    # TODO: Build DAG logic
    return None


# ============================================================
# BOSS MATCHERS
# ============================================================

def match_mha(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    if node.op != "call_function" or node.target != Fusion.fused_attention:
        return None
    

    _MHA_TAIL_PATTERN: Tuple[PatternItem, ...] = (
        (Fusion.fused_attention,),
        ("view",),
        ("transpose",),
        ("contiguous",),
        ("view",),
    )
    
    return None

def match_feedforward(ctx: FusionContext, node: fx.Node, pattern: Tuple[PatternItem, ...]) -> Optional[FusionSpec]:
    # TODO: Build DAG logic
    return None