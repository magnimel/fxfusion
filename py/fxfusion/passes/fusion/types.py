import torch
import torch.fx as fx

from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


PatternAtom = Union[Type[Any], Any]
PatternItem = Tuple[PatternAtom, ...]


class PassLevel(IntEnum):
    ATOM = 10        # Primitive engine ops
    MOLECULE = 20    # Transformer subgraph ops
    BOSS = 30        # Large fused kernels


@dataclass
class FusionContext:
    fx_model: fx.GraphModule
    graph: fx.Graph
    modules: Dict[str, Any]


@dataclass
class FusionSpec:
    tail_node: fx.Node
    primary_node: fx.Node
    input_args: Tuple[Any, ...]
    nodes_to_erase: List[fx.Node]
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, Any]] = None
    replace_nodes: Optional[List[fx.Node]] = None
    buffer_name: Optional[str] = None


MatcherFn = Callable[
    [FusionContext, fx.Node, Tuple[PatternItem, ...]],
    Optional[FusionSpec],
]