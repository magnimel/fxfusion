import torch
import torch.nn as nn
import torch.fx as fx
from typing import Any, Dict, Optional, List

from fxfusion.passes.fusion.ops import FUSION_REGISTRY, FusionOp
from fxfusion.passes.fusion.types import FusionSpec, FusionContext, PassLevel

class FusionPass:
    def __init__(self, registry: Dict[int, FusionOp] = FUSION_REGISTRY) -> None:
        self.ctx: Optional[FusionContext] = None
        self.registry = registry
        self.fx_model: Optional[fx.GraphModule] = None
        self.graph: Optional[fx.Graph] = None
        self.modules: Dict[str, Any] = {}
        self.fused_count = 0

    def run(self, model: nn.Module, device: str = "cpu") -> fx.GraphModule:
        self.fx_model = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        self.graph = self.fx_model.graph
        self.modules = dict(self.fx_model.named_modules())
        self.fused_count = 0

        ctx = FusionContext(
            fx_model=self.fx_model,
            graph=self.graph,
            modules=self.modules,
            device=torch.device(device),
        )

        def _fusion_ops_for_phase(phase: PassLevel) -> list[FusionOp]:
            """Return fusion ops for a compiler phase in registry priority order."""
            return [op for op in self.registry if op.level == phase]

        for phase in (PassLevel.ATOM, PassLevel.MOLECULE, PassLevel.BOSS):
            for fusion_op in _fusion_ops_for_phase(phase):
                for node in list(self.graph.nodes):

                    spec = fusion_op.matcher(ctx, node, fusion_op.pattern)
                    if spec is None:
                        continue

                    self._commit_fusion(fusion_op, spec)

            self.graph.eliminate_dead_code()
            self.graph.lint()
            self.fx_model.recompile()

        return self.fx_model

    def _commit_fusion(self, fusion_op: FusionOp, spec: FusionSpec) -> fx.Node:
        assert self.fx_model is not None
        assert self.graph is not None

        base_name = spec.buffer_name if spec.buffer_name is not None else str(spec.primary_node.name)
        fused_args: List[Any] = list(spec.input_args)

        with self.graph.inserting_before(spec.tail_node):
            if spec.weight is not None:
                weight_attr_name = f"{base_name}_fused_weight"
                self.fx_model.register_buffer(weight_attr_name, spec.weight)
                fused_args.append(self.graph.get_attr(weight_attr_name))

            if spec.bias is not None:
                bias_attr_name = f"{base_name}_fused_bias"
                self.fx_model.register_buffer(bias_attr_name, spec.bias)
                fused_args.append(self.graph.get_attr(bias_attr_name))

            if spec.extra is not None:
                fused_args.append(spec.extra)

            fused_node = self.graph.call_function(
                fusion_op.target,
                args=tuple(fused_args),
            )

        if spec.replace_nodes is not None:
            for out_node in spec.replace_nodes:
                out_node.replace_all_uses_with(fused_node)
        else:
            spec.tail_node.replace_all_uses_with(fused_node)

        for node in spec.nodes_to_erase:
            self.graph.erase_node(node)

        self.fused_count += 1
        return fused_node