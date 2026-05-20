import torch.nn as nn
import operator
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List, Type, Union, Callable, cast

PatternItem = Union[Type[Any], Any]

@dataclass(frozen=True)
class FusionOp:
    id: int
    name: str
    target: Callable[..., None]  
    pattern: Tuple[PatternItem, ...]

class Fusion:

    @staticmethod
    def fused_linear_relu(*args) -> None:
        raise RuntimeError("Symbolic Op: fused_linear_relu should only execute in the C++ runtime engine.")

    @staticmethod
    def fused_conv2d(*args) -> None:
        raise RuntimeError("Symbolic Op: fused_conv2d should only execute in the C++ runtime engine.")

    @staticmethod
    def fused_conv2d_relu(*args) -> None:
        raise RuntimeError("Symbolic Op: fused_conv2d_relu should only execute in the C++ runtime engine.")
    
    @staticmethod
    def fused_add_relu(*args) -> None:
        raise RuntimeError("Symbolic Op: fused_add_relu should only execute in the C++ runtime engine.")

FUSION_REGISTRY: Dict[int, FusionOp] = {
    0: FusionOp(
        id=0,
        name="fused_conv2d_relu",
        target=Fusion.fused_conv2d_relu,
        pattern=(nn.Conv2d, nn.BatchNorm2d, nn.ReLU),
    ),
    1: FusionOp(
        id=1,
        name="fused_linear_relu",
        target=Fusion.fused_linear_relu,
        pattern=(nn.Linear, nn.ReLU),
    ),
    2: FusionOp(
        id=2,
        name="fused_conv2d",
        target=Fusion.fused_conv2d,
        pattern=(nn.Conv2d, nn.BatchNorm2d),
    ),
    3: FusionOp(
        id=3,
        name="fused_add_relu",
        target=Fusion.fused_add_relu,
        pattern=(operator.add, nn.ReLU),
    ),
}