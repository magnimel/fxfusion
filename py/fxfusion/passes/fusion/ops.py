import torch
import torch.nn as nn
import torch.nn.functional as F
import operator

from dataclasses import dataclass
from typing import Callable, Tuple

from fxfusion.passes.fusion.types import (
    MatcherFn,
    PassLevel,
    PatternItem,
)

from fxfusion.passes.fusion.symbols import Fusion

from fxfusion.passes.fusion.matchers import (
    match_relu,
    match_conv_bn_relu,
    match_linear_relu,
    match_conv_bn,
    match_add_relu,
    match_add_layernorm,
    match_layernorm,
    match_conv_relu,
    match_conv,
    match_linear,
    match_embedding,
    match_qkv_linear,
    match_attention,
    match_residual_add,
    match_mha,
    match_feedforward,
)


@dataclass(frozen=True)
class FusionOp:
    name: str
    target: Callable[..., None]
    matcher: MatcherFn
    pattern: Tuple[PatternItem, ...] = ()
    level: PassLevel = PassLevel.ATOM


FUSION_REGISTRY: list[FusionOp] = [

    # ============================================================
    # LEVEL 10: ATOMS
    # ============================================================

    FusionOp(
        name="relu",
        target=Fusion.relu,
        matcher=match_relu,
        pattern=((nn.ReLU, torch.relu, F.relu),),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="conv2d_bn_relu",
        target=Fusion.conv2d_relu,
        matcher=match_conv_bn_relu,
        pattern=((nn.Conv2d,), (nn.BatchNorm2d,), (Fusion.relu,)),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="linear_relu",
        target=Fusion.linear_relu,
        matcher=match_linear_relu,
        pattern=((nn.Linear,), (Fusion.relu,)),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="conv2d_bn",
        target=Fusion.conv2d,
        matcher=match_conv_bn,
        pattern=((nn.Conv2d,), (nn.BatchNorm2d,)),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="add_relu",
        target=Fusion.add_relu,
        matcher=match_add_relu,
        pattern=((operator.add,), (Fusion.relu,)),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="conv2d_relu",
        target=Fusion.conv2d_relu,
        matcher=match_conv_relu,
        pattern=((nn.Conv2d,), (Fusion.relu,)),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="conv2d",
        target=Fusion.conv2d,
        matcher=match_conv,
        pattern=((nn.Conv2d,),),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="linear",
        target=Fusion.linear,
        matcher=match_linear,
        pattern=((nn.Linear,),),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="add_layernorm",
        target=Fusion.add_layernorm,
        matcher=match_add_layernorm,
        pattern=((operator.add,), (nn.LayerNorm,)),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="layernorm",
        target=Fusion.layernorm,
        matcher=match_layernorm,
        pattern=((nn.LayerNorm,),),
        level=PassLevel.ATOM,
    ),

    FusionOp(
        name="embedding",
        target=Fusion.embedding,
        matcher=match_embedding,
        pattern=((nn.Embedding,),),
        level=PassLevel.ATOM,
    ),

    # ============================================================
    # LEVEL 20: MOLECULES
    # ============================================================

    FusionOp(
        name="qkv_linear",
        target=Fusion.qkv_linear,
        matcher=match_qkv_linear,
        pattern=(
            (Fusion.linear,),
            ("view",),
            ("transpose",),
        ),
        level=PassLevel.MOLECULE,
    ),

    FusionOp(
        name="attention",
        target=Fusion.attention,
        matcher=match_attention,
        pattern=(
            (Fusion.qkv_linear,),
            (torch.matmul,),
            (operator.truediv,),
            ("masked_fill",),
            (F.softmax,),
            (nn.Dropout,),
            (torch.matmul,),
        ),
        level=PassLevel.MOLECULE,
    ),

    FusionOp(
        name="residual_add",
        target=Fusion.residual_add,
        matcher=match_residual_add,
        level=PassLevel.MOLECULE,
    ),

    # ============================================================
    # LEVEL 30: BOSSES
    # ============================================================

    FusionOp(
        name="mha",
        target=Fusion.mha,
        matcher=match_mha,
        pattern=(
            (Fusion.attention,),
            ("transpose",),
            ("contiguous",),
            ("view",),
            (Fusion.linear,),
        ),
        level=PassLevel.BOSS,
    ),

    FusionOp(
        name="feedforward",
        target=Fusion.feedforward,
        matcher=match_feedforward,
        pattern=((Fusion.linear_relu,), (Fusion.linear,)),
        level=PassLevel.BOSS,
    ),
]