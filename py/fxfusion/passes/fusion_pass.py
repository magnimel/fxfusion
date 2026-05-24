import torch
import torch.nn as nn
import torch.fx as fx
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List, Type, Union, Callable, cast
from fxfusion.passes.fusion import FUSION_REGISTRY, FusionOp

PatternItem = Union[Type[Any], Any]

@dataclass
class FusionSpec:
    tail_node: fx.Node
    primary_node: fx.Node
    input_args: Tuple[Any, ...]
    nodes_to_erase: List[fx.Node]
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    extra: Optional[Dict[str, Any]] = None

class FusionPass:
    def __init__(self, registry: Dict[int, FusionOp] = FUSION_REGISTRY) -> None:
        self.registry = registry
        self.fx_model: Optional[fx.GraphModule] = None
        self.graph: Optional[fx.Graph] = None
        self.modules: Dict[str, Any] = {}
        self.fused_count = 0

        self.dispatch_map: Dict[int, Callable[[List[fx.Node]], FusionSpec]] = {
            0: self._build_conv_bn_relu_spec,
            1: self._build_linear_relu_spec,
            2: self._build_conv_bn_spec,
            3: self._build_add_relu_spec,
        }

    def run(self, model: nn.Module) -> fx.GraphModule:
        self.fx_model = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
        self.graph = self.fx_model.graph
        self.modules = dict(self.fx_model.named_modules())
        self.fused_count = 0

        for fusion_op in self.registry.values():
            for node in list(self.graph.nodes):
                chain = self._get_node_chain(node, len(fusion_op.pattern))

                if chain is None:
                    continue

                if not self._ismatch(fusion_op.pattern, chain):
                    continue

                if self._has_external_users(chain):
                    continue

                self._fuse(fusion_op, chain)

        self.graph.lint()
        self.fx_model.recompile()
        return self.fx_model

    def _get_node_chain(self, tail_node: fx.Node, depth: int) -> Optional[List[fx.Node]]:
        chain = [tail_node]
        curr = tail_node

        for _ in range(depth - 1):
            if not curr.args or not isinstance(curr.args[0], fx.Node):
                return None

            curr = curr.args[0]
            chain.append(curr)

        return list(reversed(chain))

    def _ismatch(self, pattern: Tuple[PatternItem, ...], chain: List[fx.Node]) -> bool:
        for expected_op, current_node in zip(pattern, chain):
            if isinstance(expected_op, type):
                if current_node.op != "call_module":
                    return False

                if not isinstance(current_node.target, str):
                    return False
                
                target_mod = self.modules.get(current_node.target)

                if target_mod is None or not isinstance(target_mod, expected_op):
                    return False

            else:
                if current_node.op != "call_function":
                    return False

                if current_node.target != expected_op:
                    return False

        return True

    def _has_external_users(self, chain: List[fx.Node]) -> bool:
        chain_set = set(chain)

        for node in chain[:-1]:
            external_users = [user for user in node.users if user not in chain_set]

            if external_users:
                return True

        return False

    def _fuse(self, fusion_op: FusionOp, chain: List[fx.Node]) -> fx.Node:
        handler = self.dispatch_map.get(fusion_op.id)

        if handler is None:
            raise NotImplementedError(f"Fusion strategy for pattern {fusion_op.id} not implemented.")

        spec = handler(chain)
        return self._commit_fusion(fusion_op, spec)

    def _build_conv_bn_relu_spec(self, chain: List[fx.Node]) -> FusionSpec:
        conv_node, bn_node, relu_node = chain

        conv_mod = cast(nn.Conv2d, self.modules[str(conv_node.target)])
        bn_mod = cast(nn.BatchNorm2d, self.modules[str(bn_node.target)])

        weight, bias = self._fuse_conv_bn_weights(
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
            nodes_to_erase=[relu_node, bn_node, conv_node],
            extra=self._conv_extra(conv_mod),
        )

    def _build_conv_bn_spec(self, chain: List[fx.Node]) -> FusionSpec:
        conv_node, bn_node = chain

        conv_mod = cast(nn.Conv2d, self.modules[str(conv_node.target)])
        bn_mod = cast(nn.BatchNorm2d, self.modules[str(bn_node.target)])

        weight, bias = self._fuse_conv_bn_weights(
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
            nodes_to_erase=[bn_node, conv_node],
            extra=self._conv_extra(conv_mod),
        )

    def _build_linear_relu_spec(self, chain: List[fx.Node]) -> FusionSpec:
        linear_node, relu_node = chain

        linear_mod = cast(nn.Linear, self.modules[str(linear_node.target)])

        weight = linear_mod.weight.detach()
        bias = linear_mod.bias.detach() if linear_mod.bias is not None else torch.zeros(
            linear_mod.out_features,
            device=weight.device,
            dtype=weight.dtype,
        )

        return FusionSpec(
            tail_node=relu_node,
            primary_node=linear_node,
            input_args=(linear_node.args[0],),
            weight=weight,
            bias=bias,
            nodes_to_erase=[relu_node, linear_node],
        )

    def _build_add_relu_spec(self, chain: List[fx.Node]) -> FusionSpec:
        add_node, relu_node = chain

        return FusionSpec(
            tail_node=relu_node,
            primary_node=add_node,
            input_args=tuple(add_node.args,),
            nodes_to_erase=[relu_node, add_node],
        )

    def _commit_fusion(self, fusion_op: FusionOp, spec: FusionSpec) -> fx.Node:
        assert self.fx_model is not None
        assert self.graph is not None

        base_name = str(spec.primary_node.name)
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

        spec.tail_node.replace_all_uses_with(fused_node)

        for node in spec.nodes_to_erase:
            self.graph.erase_node(node)

        self.fused_count += 1
        return fused_node

    @staticmethod
    def _conv_extra(conv_mod: nn.Conv2d) -> Dict[str, Any]:
        return {
            "stride": conv_mod.stride,
            "padding": conv_mod.padding,
            "dilation": conv_mod.dilation,
            "groups": conv_mod.groups,
        }

    @staticmethod
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