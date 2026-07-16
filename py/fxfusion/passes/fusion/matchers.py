import operator

import torch
import torch.nn as nn
import torch.fx as fx

from typing import Optional, List, Tuple, Dict, Any, cast

from fxfusion.passes.fusion.types import (
    FusionContext,
    FusionSpec,
    PatternItem,
)

from fxfusion.passes.fusion.symbols import Fusion


# ============================================================
# UTILITY HELPERS
# ============================================================

def _get_node_chain(
    start_node: fx.Node,
    depth: int,
    *,
    forward: bool = False,
    allow_external_users: bool = False,
) -> Optional[List[fx.Node]]:
    """Extract a chain of nodes by following args[0] (backward) or users[0] (forward)."""
    if depth < 1:
        return None

    chain: List[fx.Node] = [start_node]
    curr = start_node

    for _ in range(depth - 1):
        if forward:
            users = list(curr.users)
            if not users:
                return None
            if not allow_external_users and len(users) != 1:
                return None
            next_node = users[0]
            if not isinstance(next_node, fx.Node):
                return None
            curr = next_node
        else:
            if not curr.args or not isinstance(curr.args[0], fx.Node):
                return None
            curr = curr.args[0]

        chain.append(curr)

    if not forward:
        chain = list(reversed(chain))

    if not forward and not allow_external_users:
        chain_set = set(chain)
        for chain_node in chain[:-1]:
            for user in chain_node.users:
                if user not in chain_set:
                    return None

    return chain


def _match_node(
    ctx: FusionContext,
    expected_op,
    current_node: fx.Node,
) -> bool:
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
    start_node: fx.Node,
    pattern: Tuple[PatternItem, ...],
    *,
    forward: bool = False,
    allow_external_users: bool = False,
) -> Optional[List[fx.Node]]:
    chain = _get_node_chain(
        start_node=start_node,
        depth=len(pattern),
        forward=forward,
        allow_external_users=allow_external_users,
    )

    if chain is None:
        return None

    for expected_ops, current_node in zip(pattern, chain):
        if not any(_match_node(ctx, op, current_node) for op in expected_ops):
            return None

    return chain


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
    target_dtype = conv_w.dtype
    target_device = conv_w.device

    bn_rm = bn_rm.to(device=target_device, dtype=target_dtype)
    bn_rv = bn_rv.to(device=target_device, dtype=target_dtype)

    if conv_b is None:
        conv_b = torch.zeros_like(
            bn_rm,
            dtype=target_dtype,
            device=target_device,
        )
    else:
        conv_b = conv_b.to(device=target_device, dtype=target_dtype)

    if bn_w is None:
        bn_w = torch.ones_like(
            bn_rm,
            dtype=target_dtype,
            device=target_device,
        )
    else:
        bn_w = bn_w.to(device=target_device, dtype=target_dtype)

    if bn_b is None:
        bn_b = torch.zeros_like(
            bn_rm,
            dtype=target_dtype,
            device=target_device,
        )
    else:
        bn_b = bn_b.to(device=target_device, dtype=target_dtype)

    scale = bn_w / torch.sqrt(bn_rv + bn_eps)
    fused_weight = conv_w * scale.view(-1, 1, 1, 1)
    fused_bias = (conv_b - bn_rm) * scale + bn_b

    return (
        fused_weight.to(dtype=target_dtype, device=target_device).detach(),
        fused_bias.to(dtype=target_dtype, device=target_device).detach(),
    )


def _fetch_attr(
    ctx: FusionContext,
    get_attr_node: fx.Node,
) -> torch.Tensor:
    attr = ctx.fx_model

    for atom in str(get_attr_node.target).split("."):
        if not hasattr(attr, atom):
            raise RuntimeError(f"Nonexistent attr: {get_attr_node.target}")

        attr = getattr(attr, atom)

    return attr


# ============================================================
# ATOM MATCHERS
# ============================================================

def match_relu(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    relu_node, = chain

    return FusionSpec(
        tail_node=relu_node,
        primary_node=relu_node,
        input_args=(relu_node.args[0],),
        weight=None,
        bias=None,
        nodes_to_erase=[relu_node],
        replace_nodes=[relu_node],
    )


def match_conv_bn_relu(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    conv_node, bn_node, relu_node = chain

    conv_mod = cast(nn.Conv2d, ctx.modules[str(conv_node.target)])
    bn_mod = cast(nn.BatchNorm2d, ctx.modules[str(bn_node.target)])

    weight, bias = _fuse_conv_bn_weights(
        conv_mod.weight,
        conv_mod.bias,
        bn_mod.running_mean,
        bn_mod.running_var,
        bn_mod.eps,
        bn_mod.weight,
        bn_mod.bias,
    )

    return FusionSpec(
        tail_node=relu_node,
        primary_node=conv_node,
        input_args=(conv_node.args[0],),
        weight=weight,
        bias=bias,
        nodes_to_erase=[
            relu_node,
            bn_node,
            conv_node,
        ],
        extra=_conv_extra(conv_mod),
    )


def match_linear_relu(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    linear_node, relu_node = chain
    linear_mod = cast(nn.Linear, ctx.modules[str(linear_node.target)])

    weight = linear_mod.weight.detach()
    bias = (
        linear_mod.bias.detach()
        if linear_mod.bias is not None
        else torch.zeros(
            linear_mod.out_features,
            device=weight.device,
            dtype=weight.dtype,
        )
    )

    return FusionSpec(
        tail_node=relu_node,
        primary_node=linear_node,
        input_args=(linear_node.args[0],),
        weight=weight,
        bias=bias,
        nodes_to_erase=[
            relu_node,
            linear_node,
        ],
    )


def match_conv_bn(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    conv_node, bn_node = chain

    conv_mod = cast(nn.Conv2d, ctx.modules[str(conv_node.target)])
    bn_mod = cast(nn.BatchNorm2d, ctx.modules[str(bn_node.target)])

    weight, bias = _fuse_conv_bn_weights(
        conv_mod.weight,
        conv_mod.bias,
        bn_mod.running_mean,
        bn_mod.running_var,
        bn_mod.eps,
        bn_mod.weight,
        bn_mod.bias,
    )

    return FusionSpec(
        tail_node=bn_node,
        primary_node=conv_node,
        input_args=(conv_node.args[0],),
        weight=weight,
        bias=bias,
        nodes_to_erase=[
            bn_node,
            conv_node,
        ],
        extra=_conv_extra(conv_mod),
    )


def match_add_relu(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    add_node, relu_node = chain

    return FusionSpec(
        tail_node=relu_node,
        primary_node=add_node,
        input_args=tuple(add_node.args),
        nodes_to_erase=[
            relu_node,
            add_node,
        ],
    )


def match_conv_relu(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    conv_node, relu_node = chain
    conv_mod = cast(nn.Conv2d, ctx.modules[str(conv_node.target)])

    weight = conv_mod.weight.detach()
    bias = (
        conv_mod.bias.detach()
        if conv_mod.bias is not None
        else torch.zeros(
            conv_mod.out_channels,
            device=weight.device,
            dtype=weight.dtype,
        )
    )

    return FusionSpec(
        tail_node=relu_node,
        primary_node=conv_node,
        input_args=(conv_node.args[0],),
        weight=weight,
        bias=bias,
        nodes_to_erase=[
            relu_node,
            conv_node,
        ],
        extra=_conv_extra(conv_mod),
    )


def match_conv(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    conv_node, = chain
    conv_mod = cast(nn.Conv2d, ctx.modules[str(conv_node.target)])

    weight = conv_mod.weight.detach()
    bias = (
        conv_mod.bias.detach()
        if conv_mod.bias is not None
        else torch.zeros(
            conv_mod.out_channels,
            device=weight.device,
            dtype=weight.dtype,
        )
    )

    return FusionSpec(
        tail_node=conv_node,
        primary_node=conv_node,
        input_args=(conv_node.args[0],),
        weight=weight,
        bias=bias,
        nodes_to_erase=[conv_node],
        extra=_conv_extra(conv_mod),
    )


def match_linear(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    linear_node, = chain
    linear_mod = cast(nn.Linear, ctx.modules[str(linear_node.target)])

    weight = linear_mod.weight.detach()
    bias = (
        linear_mod.bias.detach()
        if linear_mod.bias is not None
        else torch.zeros(
            linear_mod.out_features,
            device=weight.device,
            dtype=weight.dtype,
        )
    )

    return FusionSpec(
        tail_node=linear_node,
        primary_node=linear_node,
        input_args=(linear_node.args[0],),
        weight=weight,
        bias=bias,
        nodes_to_erase=[linear_node],
    )

def match_layernorm(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    norm_node, = chain
    norm_mod = cast(nn.LayerNorm, ctx.modules[str(norm_node.target)])

    normalized_shape = tuple(norm_mod.normalized_shape)

    weight = (
        norm_mod.weight.detach()
        if norm_mod.weight is not None
        else torch.ones(
            normalized_shape,
            dtype=torch.float32,
            device=ctx.device,
        )
    )

    bias = (
        norm_mod.bias.detach()
        if norm_mod.bias is not None
        else torch.zeros(
            normalized_shape,
            dtype=weight.dtype,
            device=weight.device,
        )
    )

    extra = {
        "normalized_shape": list(normalized_shape),
        "eps": norm_mod.eps,
    }

    return FusionSpec(
        tail_node=norm_node,
        primary_node=norm_node,
        input_args=(norm_node.args[0],),
        weight=weight,
        bias=bias,
        nodes_to_erase=[norm_node],
        extra=extra,
    )
    
    
def match_add_layernorm(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    add_node, norm_node = chain
    norm_mod = cast(nn.LayerNorm, ctx.modules[str(norm_node.target)])

    normalized_shape = tuple(norm_mod.normalized_shape)

    weight = (
        norm_mod.weight.detach()
        if norm_mod.weight is not None
        else torch.ones(
            normalized_shape,
            dtype=torch.float32,
            device=ctx.device,
        )
    )

    bias = (
        norm_mod.bias.detach()
        if norm_mod.bias is not None
        else torch.zeros(
            normalized_shape,
            dtype=weight.dtype,
            device=weight.device,
        )
    )

    extra = {
        "normalized_shape": list(normalized_shape),
        "eps": norm_mod.eps,
    }

    return FusionSpec(
        tail_node=norm_node,
        primary_node=add_node,
        input_args=tuple(add_node.args),
        weight=weight,
        bias=bias,
        nodes_to_erase=[
            norm_node,
            add_node,
        ],
        extra=extra,
    )


def match_embedding(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
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
# MOLECULE MATCHERS
# ============================================================

def match_qkv_linear(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    if node.op != "call_function" or node.target != Fusion.linear:
        return None

    input_node = node.args[0]

    if not isinstance(input_node, fx.Node):
        return None

    linear_siblings = [
        user for user in input_node.users
        if user.op == "call_function" and user.target == Fusion.linear
    ]

    if len(linear_siblings) != 3:
        return None

    linear_siblings = sorted(linear_siblings, key=lambda n: n.name)

    if node is not linear_siblings[0]:
        return None

    q_node, k_node, v_node = linear_siblings

    q_chain = _match_chain_pattern(ctx, q_node, pattern, forward=True)
    k_chain = _match_chain_pattern(ctx, k_node, pattern, forward=True)
    v_chain = _match_chain_pattern(ctx, v_node, pattern, forward=True)

    if q_chain is None or k_chain is None or v_chain is None:
        return None

    q_weight_node = q_node.args[1]
    q_weight_name = str(q_weight_node.target)
    qkv_base_name = q_weight_name.replace("_w_q_fused_weight", "_qkv")

    packed_weight = torch.cat(
        [
            _fetch_attr(ctx, q_node.args[1]),
            _fetch_attr(ctx, k_node.args[1]),
            _fetch_attr(ctx, v_node.args[1]),
        ],
        dim=0,
    )

    packed_bias = torch.cat(
        [
            _fetch_attr(ctx, q_node.args[2]),
            _fetch_attr(ctx, k_node.args[2]),
            _fetch_attr(ctx, v_node.args[2]),
        ],
        dim=0,
    )

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

            q_node.args[1],
            q_node.args[2],
            k_node.args[1],
            k_node.args[2],
            v_node.args[1],
            v_node.args[2],
        ],
        replace_nodes=[
            q_chain[2],
            k_chain[2],
            v_chain[2],
        ],
        buffer_name=qkv_base_name,
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
    if node.op != "call_function" or node.target != Fusion.qkv_linear:
        return None

    qkv_node = node

    qkv_input = qkv_node.args[0]
    qkv_weight_node = qkv_node.args[1]
    qkv_bias_node = qkv_node.args[2]
    qkv_extra = qkv_node.args[3]

    kt_users = [
        user for user in qkv_node.users
        if user.op == "call_method" and user.target == "transpose"
    ]

    if len(kt_users) != 1:
        return None

    kt_node = kt_users[0]

    qk_users = [
        user for user in kt_node.users
        if user.op == "call_function" and user.target == torch.matmul
    ]

    if len(qk_users) != 1:
        return None

    qk_matmul_node = qk_users[0]

    if qk_matmul_node.args[0] is not qkv_node:
        return None

    if qk_matmul_node.args[1] is not kt_node:
        return None

    attn_chain = _match_chain_pattern(
        ctx,
        qk_matmul_node,
        pattern[1:],
        forward=True,
    )

    if attn_chain is None:
        return None

    (
        qk_matmul_node,
        scale_node,
        masked_fill_node,
        softmax_node,
        dropout_node,
        av_matmul_node,
    ) = attn_chain

    eq_node = masked_fill_node.args[1]

    if not isinstance(eq_node, fx.Node):
        return None

    if eq_node.op != "call_function" or eq_node.target != operator.eq:
        return None

    mask_input = eq_node.args[0]

    if not isinstance(mask_input, fx.Node):
        return None

    if av_matmul_node.args[0] is not dropout_node:
        return None

    if av_matmul_node.args[1] is not qkv_node:
        return None

    attn_extra = dict(qkv_extra)
    attn_extra["scale_divisor"] = scale_node.args[1]

    return FusionSpec(
        tail_node=av_matmul_node,
        primary_node=qkv_node,
        input_args=(
            qkv_input,
            mask_input,
            qkv_weight_node,
            qkv_bias_node,
        ),
        weight=None,
        bias=None,
        nodes_to_erase=[
            av_matmul_node,
            dropout_node,
            softmax_node,
            masked_fill_node,
            eq_node,
            scale_node,
            qk_matmul_node,
            kt_node,
            qkv_node,
        ],
        replace_nodes=[av_matmul_node],
        extra=attn_extra,
    )


def match_residual_add(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    # TODO: Build DAG logic.
    return None


# ============================================================
# BOSS MATCHERS
# ============================================================

def match_mha(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    (
        attn_node,
        out_transpose_node,
        contiguous_node,
        view_node,
        linear_node,
    ) = chain

    qkv_input = attn_node.args[0]
    mask_input = attn_node.args[1]
    qkv_weight_node = attn_node.args[2]
    qkv_bias_node = attn_node.args[3]
    attn_extra = attn_node.args[4]

    o_weight_node = linear_node.args[1]
    o_bias_node = linear_node.args[2]

    return FusionSpec(
        tail_node=linear_node,
        primary_node=linear_node,
        input_args=(
            qkv_input,
            mask_input,
            qkv_weight_node,
            qkv_bias_node,
            o_weight_node,
            o_bias_node,
        ),
        weight=None,
        bias=None,
        nodes_to_erase=[
            linear_node,
            view_node,
            contiguous_node,
            out_transpose_node,
            attn_node,
        ],
        replace_nodes=[linear_node],
        extra=attn_extra,
    )


def match_feedforward(
    ctx: FusionContext,
    node: fx.Node,
    pattern: Tuple[PatternItem, ...],
) -> Optional[FusionSpec]:
    chain = _match_chain_pattern(ctx, node, pattern)

    if chain is None:
        return None

    linear_relu_node, linear_node = chain

    linear_relu_weight = linear_relu_node.args[1]
    linear_relu_bias = linear_relu_node.args[2]
    linear_weight = linear_node.args[1]
    linear_bias = linear_node.args[2]

    return FusionSpec(
        tail_node=linear_node,
        primary_node=linear_relu_node,
        input_args=(
            linear_relu_node.args[0],
            linear_relu_weight,
            linear_relu_bias,
            linear_weight,
            linear_bias,
        ),
        weight=None,
        bias=None,
        nodes_to_erase=[
            linear_node,
            linear_relu_node,
        ],
        replace_nodes=[linear_node],
    )