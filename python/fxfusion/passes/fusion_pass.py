import torch
import torch.nn as nn
import torch.fx as fx
from typing import Type, Dict, Any, Tuple, Optional, List, cast
from fxfusion.passes.fusion import FUSION_REGISTRY, FusionOp


class FusionPass:

    def __init__(self, registry: Dict[int, FusionOp] = FUSION_REGISTRY) -> None:
        self.registry = registry
        self.fx_model: Optional[fx.GraphModule] = None
        self.graph: Optional[fx.Graph] = None
        self.modules: Dict[str, Any] = {}
        self.fused_count: int = 0

    def run(self, model: nn.Module) -> fx.GraphModule:
        """Entry point to execute the transformation pass."""
        # Ensure we are working with an FX GraphModule
        if not isinstance(model, fx.GraphModule):
            self.fx_model = fx.symbolic_trace(model)
        else:
            self.fx_model = model
            
        self.graph = self.fx_model.graph
        self.modules = dict(self.fx_model.named_modules())
        self.fused_count = 0

        # Execute lowering passes against the registered patterns
        for pattern_id, fusion_op in self.registry.items():
            for node in list(self.graph.nodes):
                if self._matches_module_pattern(fusion_op.pattern, node):
                    if self._has_external_users(fusion_op.pattern, node):
                        continue
                    
                    self._fuse(fusion_op, node)

        self.graph.lint()
        self.fx_model.recompile()
        return self.fx_model

    def _matches_module_pattern(self, pattern: Tuple[Type[nn.Module], ...], node: fx.Node) -> bool:
        """Determines routing logic based on pattern depth."""
        if len(pattern) == 3:
            return self._pattern3(pattern, node)
        elif len(pattern) == 2:
            return self._pattern2(pattern, node)
        return False

    def _pattern3(self, pattern: Tuple[Type[nn.Module], ...], node: fx.Node) -> bool:
        if len(node.args) == 0 or not isinstance(node.args[0], fx.Node):
            return False
        parent = node.args[0]
        if len(parent.args) == 0 or not isinstance(parent.args[0], fx.Node):
            return False
            
        nodes = (parent.args[0], parent, node)
        return self._ismatch(pattern, nodes)

    def _pattern2(self, pattern: Tuple[Type[nn.Module], ...], node: fx.Node) -> bool:
        if len(node.args) == 0 or not isinstance(node.args[0], fx.Node):
            return False
        nodes = (node.args[0], node)
        return self._ismatch(pattern, nodes)

    def _ismatch(self, pattern: Tuple[Type[nn.Module], ...], nodes: Tuple[fx.Node, ...]) -> bool:
        """Validates operator identity against registered target modules."""
        for expected_type, current_node in zip(pattern, nodes):
            if current_node.op != 'call_module' or not isinstance(current_node.target, str):
                return False
            if current_node.target not in self.modules:
                return False
            if not isinstance(self.modules[current_node.target], expected_type):
                return False
        return True

    def _has_external_users(self, pattern: Tuple[Type[nn.Module], ...], tail_node: fx.Node) -> bool:
        """Ensures intermediate nodes do not leak execution activations downstream."""
        if len(pattern) == 3:
            secondary_node = cast(fx.Node, tail_node.args[0])
            primary_node = cast(fx.Node, secondary_node.args[0])
            return len(secondary_node.users) > 1 or len(primary_node.users) > 1
            
        elif len(pattern) == 2:
            primary_node = cast(fx.Node, tail_node.args[0])
            return len(primary_node.users) > 1
            
        return False

    def _fuse(self, fusion_op: FusionOp, node: fx.Node) -> fx.Node:
        """Phase 1 Strategy Router: Extracts sub-graph tensors strictly via data dependencies."""
        if fusion_op.id == 0:  # (Conv2d, BatchNorm2d, ReLU)
            bn_node = cast(fx.Node, node.args[0])
            primary_node = cast(fx.Node, bn_node.args[0]) # The root Conv2d node
            input_node = cast(fx.Node, primary_node.args[0])
            
            nodes_to_erase = [node, bn_node, primary_node]
            
            conv_mod = cast(nn.Conv2d, self.modules[str(primary_node.target)])
            bn_mod = cast(nn.BatchNorm2d, self.modules[str(bn_node.target)])
            
            extra = {
                "stride": conv_mod.stride,
                "padding": conv_mod.padding,
                "dilation": conv_mod.dilation,
                "groups": conv_mod.groups,
            }
            
            weight, bias = self._fuse_conv_bn_weights(
                conv_mod.weight, conv_mod.bias,
                bn_mod.running_mean, bn_mod.running_var, bn_mod.eps,
                bn_mod.weight, bn_mod.bias
            )

        elif fusion_op.id == 1:  # (Linear, ReLU)
            primary_node = cast(fx.Node, node.args[0])
            input_node = cast(fx.Node, primary_node.args[0])
            nodes_to_erase = [node, primary_node]
            
            linear_mod = cast(nn.Linear, self.modules[str(primary_node.target)])
            weight, bias = linear_mod.weight.detach(), cast(Optional[torch.Tensor], linear_mod.bias)
            if bias is None:
                bias = torch.zeros(linear_mod.out_features, device=weight.device, dtype=weight.dtype)
            extra = None
            
        elif fusion_op.id == 2:  # (Conv2d, BatchNorm2d)
            primary_node = cast(fx.Node, node.args[0])
            input_node = cast(fx.Node, primary_node.args[0])
            nodes_to_erase = [node, primary_node]
            
            conv_mod = cast(nn.Conv2d, self.modules[str(primary_node.target)])
            bn_mod = cast(nn.BatchNorm2d, self.modules[str(node.target)])
            
            extra = {
                "stride": conv_mod.stride,
                "padding": conv_mod.padding,
                "dilation": conv_mod.dilation,
                "groups": conv_mod.groups,
            }
            
            
            weight, bias = self._fuse_conv_bn_weights(
                conv_mod.weight, conv_mod.bias,
                bn_mod.running_mean, bn_mod.running_var, bn_mod.eps,
                bn_mod.weight, bn_mod.bias
            )
        else:
            raise NotImplementedError(f"Fusion strategy for pattern {fusion_op.id} not implemented.")

        return self._commit_fusion(
            fusion_op=fusion_op,
            tail_node=node,
            primary_node=primary_node,
            input_node=input_node,
            weight=weight,
            bias=bias,
            nodes_to_erase=nodes_to_erase,
            extra=extra
        )

    def _commit_fusion(
        self,
        fusion_op: FusionOp,
        tail_node: fx.Node,
        primary_node: fx.Node,
        input_node: fx.Node,
        weight: torch.Tensor,
        bias: torch.Tensor,
        nodes_to_erase: List[fx.Node],
        extra: Optional[Dict[str, Any]]
    ) -> fx.Node:
        """Handles Phase 2: Commits persistent buffers, rewires IR edges, and sweeps stale nodes."""
        assert self.fx_model is not None and self.graph is not None
        
        # Using base_name extracted directly from the native execution node identifier
        base_name = str(primary_node.name)
        
        weight_attr_name = f"{base_name}_fused_weight"
        bias_attr_name = f"{base_name}_fused_bias"
        
        self.fx_model.register_buffer(weight_attr_name, weight)
        self.fx_model.register_buffer(bias_attr_name, bias)
        
        with self.graph.inserting_before(tail_node):
            weight_node = self.graph.get_attr(weight_attr_name)
            bias_node = self.graph.get_attr(bias_attr_name)
        
            fused_node = self.graph.call_function(
                fusion_op.target,
                args=(input_node, weight_node, bias_node, extra)
            ) 
                
        tail_node.replace_all_uses_with(fused_node)
        
        for n in nodes_to_erase:
            self.graph.erase_node(n)
            
        self.fused_count += 1
        return fused_node

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