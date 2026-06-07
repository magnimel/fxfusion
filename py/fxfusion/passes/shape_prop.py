import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.functional as F
from typing import Type, Dict, Any, Tuple, Optional, List, cast
from fxfusion.passes.fusion import Fusion
from torch.fx.node import map_arg

class ShapePropPass():    
    
    def __init__(self, fx_model: fx.GraphModule) -> None:
        self.fx_model : fx.GraphModule = fx_model
        self.graph : fx.Graph = self.fx_model.graph
        self.modules: Dict[str, Any] = dict(self.fx_model.named_modules())
    
    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, object] = {}

        def _load_arg(a: fx.Node):
            return map_arg(a, lambda n: env[n.name])

        def _fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.fx_model
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            result: Optional[object] = None
            
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = _fetch_attr(node.target)
            elif node.op == 'call_function':
                if node.target in (Fusion.fused_conv2d_relu, Fusion.fused_conv2d):
                    x, weight, bias, extra = _load_arg(node.args)
                    
                    node.meta["attrs"] = {
                        "stride": extra["stride"],
                        "padding": extra["padding"],
                        "dilation": extra["dilation"],
                        "groups": extra["groups"]
                    }
                    
                    result = F.conv2d(input=x, weight=weight, bias=bias,
                        stride=extra["stride"],
                        padding=extra["padding"],
                        dilation=extra["dilation"],
                        groups=extra["groups"]
                    )
                elif node.target in (Fusion.fused_linear_relu, Fusion.fused_linear):
                    x, weight, bias = _load_arg(node.args)
                    result = F.linear(input=x, weight=weight, bias=bias)
                elif node.target == Fusion.fused_add_relu:
                    a, b = _load_arg(node.args)
                    result = a + b
                else:
                    result = node.target(*_load_arg(node.args), **_load_arg(node.kwargs)) 
            elif node.op == 'call_method':
                self_obj, *args = _load_arg(node.args)
                result = getattr(self_obj, node.target)(*args, **_load_arg(node.kwargs))
            elif node.op == 'call_module':
                result = self.modules[node.target](*_load_arg(node.args), **_load_arg(node.kwargs))
            elif node.op == 'output':
                result = _load_arg(node.args[0])
            else:
                raise RuntimeError(f"Unsupported node op: {node.op}")

            if isinstance(result, torch.Tensor):
                node.meta['shape'] = tuple(result.shape)
                node.meta['dtype'] = result.dtype
                node.meta['device'] = result.device

            env[node.name] = result
            
        return _load_arg(self.graph.output)