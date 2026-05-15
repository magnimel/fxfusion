import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Type


@dataclass(frozen=True)
class FusionOp:
    id: int
    name: str
    target: Callable[..., None]  
    pattern: Tuple[Type[nn.Module], ...]

class Fusion:

    @staticmethod
    def fused_linear_relu(*args) -> None:
        raise RuntimeError("Symbolic Op: fused_linear_relu should only execute in the C++ runtime engine.")

    @staticmethod
    def fused_conv2d(*args) -> None:
        raise RuntimeError("Symbolic Op: fused_conv_norm_relu should only execute in the C++ runtime engine.")

    @staticmethod
    def fused_conv2d_relu(*args) -> None:
        raise RuntimeError("Symbolic Op: fused_conv_norm_relu should only execute in the C++ runtime engine.")

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
}