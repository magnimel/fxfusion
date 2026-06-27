import torch
import torch.nn as nn
import torch.fx as fx

from dataclasses import dataclass, field
from typing import Dict, Optional, List

from fxfusion.passes.fusion.symbols import Fusion
from fxfusion.passes.memory_manager import MemoryManager


@dataclass
class TensorAlloc:
    node_name: str
    size_bytes: int
    mem_offset: Optional[int]
    kind: str  # "input", "const", "activation", "output", "alias"
    alias_of: Optional[str] = None


@dataclass
class MemoryPlan:
    spec_dict: Dict[str, TensorAlloc] = field(default_factory=dict)
    arena_size: int = 0


class MemoryPlanningPass:

    def __init__(self, fx_model: fx.GraphModule, alignment: int = 64) -> None:
        self.fx_model: fx.GraphModule = fx_model
        self.graph: fx.Graph = self.fx_model.graph
        self.memory_manager: MemoryManager = MemoryManager(alignment=alignment)
        self.modules: Dict[str, nn.Module] = dict(self.fx_model.named_modules())

    def _is_safe_unary_alias(
        self,
        node: fx.Node,
        *,
        allow_external_alias: bool = False,
    ) -> bool:
        if not node.args:
            return False

        x = node.args[0]

        if not isinstance(x, fx.Node):
            return False

        # If the input has multiple users, reusing its storage for this output
        # can break lifetime correctness unless we track alias groups.
        if len(x.users) != 1:
            return False

        # For destructive-style aliases such as relu, do not alias
        # external inputs/constants. We do not want to mutate user input
        # or model constants.
        if not allow_external_alias and x.op in ("placeholder", "get_attr"):
            return False

        return True

    def _is_view_like_alias(self, node: fx.Node) -> bool:
        if node.op == "call_function" and node.target in (
            torch.flatten,
            torch.reshape,
            torch.narrow,
        ):
            return self._is_safe_unary_alias(node, allow_external_alias=True)
        if node.op == "call_method" and node.target in ("view", "reshape", "flatten", "contiguous"):
            return self._is_safe_unary_alias(node, allow_external_alias=True)
        return False

    def _is_relu_alias(self, node: fx.Node) -> bool:
        if node.op != "call_function" or node.target != Fusion.relu:
            return False
        return self._is_safe_unary_alias(node, allow_external_alias=False)

    def _is_eval_dropout_alias(self, node: fx.Node) -> bool:
        if node.op != "call_module":
            return False
        mod = self.modules.get(str(node.target))
        if not isinstance(mod, nn.Dropout):
            return False
        if mod.training:
            return False
        return self._is_safe_unary_alias(node, allow_external_alias=True)

    def _node_kind(self, node: fx.Node) -> str:
        if self._is_view_like_alias(node):
            return "alias"
        if self._is_relu_alias(node):
            return "alias"
        if self._is_eval_dropout_alias(node):
            return "alias"
        if node.op == "placeholder":
            return "input"
        if node.op == "get_attr":
            return "const"
        if node.op in ("call_function", "call_method", "call_module"):
            return "activation"
        if node.op == "output":
            return "output"
        raise RuntimeError(f"Unsupported node op: {node.op}")

    def _compute_last_use(self, nodes: List[fx.Node]) -> Dict[fx.Node, int]:
        last_use: Dict[fx.Node, int] = {}

        for i, node in enumerate(nodes):
            for input_node in node.all_input_nodes:
                last_use[input_node] = i

        return last_use

    def _release_dead_inputs(
        self,
        node: fx.Node,
        exec_id: int,
        last_use: Dict[fx.Node, int],
    ) -> None:
        released = set()

        for arg in node.all_input_nodes:
            if last_use.get(arg) != exec_id:
                continue

            arg_kind = arg.meta.get("kind")

            if arg_kind not in ("activation", "alias"):
                continue

            arg_alloc: Optional[TensorAlloc] = arg.meta.get("alloc")

            if arg_alloc is None or arg_alloc.mem_offset is None:
                continue

            key = (arg_alloc.mem_offset, arg_alloc.size_bytes)

            if key in released:
                continue

            self.memory_manager.release(
                arg_alloc.mem_offset,
                arg_alloc.size_bytes,
            )

            released.add(key)

    def run(self) -> MemoryPlan:
        plan = MemoryPlan()
        nodes = list(self.graph.nodes)
        last_use = self._compute_last_use(nodes)

        for exec_id, node in enumerate(nodes):
            kind = self._node_kind(node)
            node.meta["kind"] = kind

            try:
                size = self.memory_manager.tensor_size_bytes(node)
            except RuntimeError:
                size = 0

            if kind in ("input", "const"):
                alloc = TensorAlloc(
                    node_name=node.name,
                    size_bytes=size,
                    mem_offset=None,
                    kind=kind,
                )

            elif kind == "activation":
                offset = self.memory_manager.allocate(size)

                alloc = TensorAlloc(
                    node_name=node.name,
                    size_bytes=size,
                    mem_offset=offset,
                    kind=kind,
                )

            elif kind in ("alias", "output"):
                returned_node = node.args[0]

                if isinstance(returned_node, fx.Node):
                    base_alloc: TensorAlloc = returned_node.meta["alloc"]

                    alloc = TensorAlloc(
                        node_name=node.name,
                        size_bytes=size,
                        mem_offset=base_alloc.mem_offset,
                        kind=kind,
                        alias_of=returned_node.name,
                    )
                else:
                    alloc = TensorAlloc(
                        node_name=node.name,
                        size_bytes=0,
                        mem_offset=None,
                        kind=kind,
                    )

            else:
                raise RuntimeError(f"Unsupported node kind: {kind}")

            node.meta["alloc"] = alloc
            node.meta["id"] = exec_id
            plan.spec_dict[node.name] = alloc

            if kind not in ("alias", "output"):
                self._release_dead_inputs(node, exec_id, last_use)

        plan.arena_size = self.memory_manager.peak_arena_top
        self.fx_model.meta["arena_size"] = plan.arena_size

        return plan


def print_alloc(fx_model: fx.GraphModule):
    print(f"\nARENA SIZE: {fx_model.meta['arena_size']}")
    print(
        f"{'Node Name':<45} | "
        f"{'Kind':<12} | "
        f"{'Size (B)':<10} | "
        f"{'Offset':<10} | "
        f"{'Alias Of'}"
    )
    print("-" * 100)

    for node in fx_model.graph.nodes:
        if "alloc" in node.meta:
            alloc = node.meta["alloc"]
            offset_val = (
                str(alloc.mem_offset) if alloc.mem_offset is not None else "N/A"
            )
            alias_of = str(alloc.alias_of) if alloc.alias_of is not None else "N/A"

            print(
                f"{node.name:<45} | "
                f"{alloc.kind:<12} | "
                f"{alloc.size_bytes:<10} | "
                f"{offset_val:<10} | "
                f"{alias_of}"
            )
        else:
            print(f"{node.name:<45} | {'Missing Alloc Info'}")
