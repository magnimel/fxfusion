from platform import node

import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.functional as F

from typing import Any, Callable, Dict, Optional
from torch.fx.node import map_arg

from fxfusion.passes.fusion.symbols import Fusion


class ShapePropPass:

    def __init__(self, fx_model: fx.GraphModule) -> None:
        self.fx_model: fx.GraphModule = fx_model
        self.graph: fx.Graph = self.fx_model.graph
        self.modules: Dict[str, Any] = dict(self.fx_model.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, object] = {}
        output_result: Optional[object] = None

        def _load_arg(a):
            return map_arg(a, lambda n: env[n.name])

        for node in self.graph.nodes:
            result: Optional[object] = None

            if node.op == "placeholder":
                try:
                    result = next(args_iter)
                except StopIteration:
                    if len(node.args) > 0:
                        result = node.args[0]
                    else:
                        raise RuntimeError(
                            f"Missing input value for placeholder: {node.name}"
                        )

            elif node.op == "get_attr":
                result = self._fetch_attr(str(node.target))

            elif node.op == "call_function":
                result = self._propagate_call_function(node, _load_arg)

            elif node.op == "call_method":
                self_obj, *method_args = _load_arg(node.args)

                result = getattr(self_obj, node.target)(
                    *method_args,
                    **_load_arg(node.kwargs),
                )

            elif node.op == "call_module":
                result = self.modules[str(node.target)](
                    *_load_arg(node.args),
                    **_load_arg(node.kwargs),
                )

            elif node.op == "output":
                result = _load_arg(node.args[0])
                output_result = result

            else:
                raise RuntimeError(f"Unsupported node op: {node.op}")

            self._record_meta(node, result)
            env[node.name] = result

        return output_result

    def _fetch_attr(self, target: str):
        target_atoms = target.split(".")
        attr_itr = self.fx_model

        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)

        return attr_itr

    def _record_meta(self, node: fx.Node, result: object) -> None:
        def scalar_device_from_inputs() -> torch.device:
            for input_node in node.all_input_nodes:
                device = input_node.meta.get("device")

                if isinstance(device, torch.device):
                    return device

            raise RuntimeError(
                f"Cannot infer scalar device for node {node.name}. "
                "No input node has device metadata."
            )

        if result is None:
            raise RuntimeError(f"ShapePropPass cannot handle None result for node {node.name}")

        if isinstance(result, torch.Tensor):
            node.meta["shape"] = tuple(result.shape)
            node.meta["dtype"] = result.dtype
            node.meta["device"] = result.device
            return
        
        if isinstance(result, torch.Size):
            node.meta["shape"] = (len(result),)
            node.meta["dtype"] = torch.int64
            node.meta["device"] = scalar_device_from_inputs()
            return

        if isinstance(result, bool):
            scalar = torch.tensor(
                result,
                dtype=torch.bool,
                device=scalar_device_from_inputs(),
            )

        elif isinstance(result, int):
            scalar = torch.tensor(
                result,
                dtype=torch.int64,
                device=scalar_device_from_inputs(),
            )

        elif isinstance(result, float):
            scalar = torch.tensor(
                result,
                dtype=torch.float32,
                device=scalar_device_from_inputs(),
            )

        else:
            raise RuntimeError(
                f"Unsupported ShapeProp result for node {node.name}: "
                f"type={type(result).__name__}, value={result!r}"
            )

        node.meta["shape"] = tuple(scalar.shape)
        node.meta["dtype"] = scalar.dtype
        node.meta["device"] = scalar.device

    def _raise_intermediate_fusion_error(self, node: fx.Node) -> None:
        target_name = getattr(node.target, "__name__", str(node.target))

        raise RuntimeError(
            f"{target_name} reached ShapePropPass, but it is an intermediate "
            "fusion IR node. It should have been consumed by a higher-level "
            "fusion pass before shape propagation.\n"
            f"Node: {node.name}"
        )

    def _propagate_call_function(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> object:
        
        if node.target == Fusion.relu:
            return self._propagate_relu(node, load_arg)

        if node.target in (Fusion.conv2d_relu, Fusion.conv2d):
            return self._propagate_conv2d(node, load_arg)

        if node.target in (Fusion.linear_relu, Fusion.linear):
            return self._propagate_linear(node, load_arg)

        if node.target == Fusion.add_relu:
            return self._propagate_add_relu(node, load_arg)

        if node.target == Fusion.layernorm:
            return self._propagate_layernorm(node, load_arg)
        
        if node.target == Fusion.add_layernorm:
            return self._propagate_add_layernorm(node, load_arg)

        if node.target == Fusion.embedding:
            return self._propagate_embedding(node, load_arg)

        if node.target in (Fusion.qkv_linear, Fusion.attention):
            self._raise_intermediate_fusion_error(node)

        if node.target == Fusion.mha:
            return self._propagate_mha(node, load_arg)

        if node.target == Fusion.feedforward:
            return self._propagate_feedforward(node, load_arg)

        return node.target(
            *load_arg(node.args),
            **load_arg(node.kwargs),
        )

    def _propagate_relu(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> torch.Tensor:
        x, = load_arg(node.args)
        return F.relu(x)

    def _propagate_conv2d(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> torch.Tensor:
        x, weight, bias, extra = load_arg(node.args)

        node.meta["attrs"] = {
            "stride": extra["stride"],
            "padding": extra["padding"],
            "dilation": extra["dilation"],
            "groups": extra["groups"],
        }

        result = F.conv2d(
            input=x,
            weight=weight,
            bias=bias,
            stride=extra["stride"],
            padding=extra["padding"],
            dilation=extra["dilation"],
            groups=extra["groups"],
        )

        if node.target == Fusion.conv2d_relu:
            result = F.relu(result)

        return result

    def _propagate_linear(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> torch.Tensor:
        x, weight, bias = load_arg(node.args)

        result = F.linear(
            input=x,
            weight=weight,
            bias=bias,
        )

        if node.target == Fusion.linear_relu:
            result = F.relu(result)

        return result

    def _propagate_add_relu(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> torch.Tensor:
        a, b = load_arg(node.args)
        return F.relu(a + b)

    def _propagate_layernorm(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> torch.Tensor:
        a, weight, bias, extra = load_arg(node.args)

        return F.layer_norm(
            a,
            normalized_shape=extra["normalized_shape"],
            weight=weight,
            bias=bias,
            eps=extra["eps"],
        )
        
    def _propagate_add_layernorm(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> torch.Tensor:
        a, b, weight, bias, extra = load_arg(node.args)

        return F.layer_norm(
            a + b,
            normalized_shape=extra["normalized_shape"],
            weight=weight,
            bias=bias,
            eps=extra["eps"],
        )

    def _propagate_embedding(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> torch.Tensor:
        x, weight = load_arg(node.args)
        return F.embedding(x, weight)

    def _propagate_mha(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> torch.Tensor:
        
        (
            x, mask, qkv_weight, qkv_bias,
            out_weight, out_bias, extra,
        ) = load_arg(node.args)

        context = self._attention_context_from_qkv(
            x=x,
            mask=mask,
            qkv_weight=qkv_weight,
            qkv_bias=qkv_bias,
            extra=extra,
        )

        batch_size = context.shape[0]
        d_model = int(extra["d_model"])

        context = (
            context
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, d_model)
        )

        return F.linear(context, out_weight, out_bias)

    def _propagate_feedforward(self, node: fx.Node, load_arg: Callable[[Any], Any]) -> torch.Tensor:
        x, w1, b1, w2, b2 = load_arg(node.args)
        hidden = F.linear(x, w1, b1)
        hidden = F.relu(hidden)
        return F.linear(hidden, w2, b2)

    def _attention_context_from_qkv(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor,
        extra: Dict[str, Any],
    ) -> torch.Tensor:
        d_model = int(extra["d_model"])
        num_heads = int(extra["num_heads"])
        head_dim = int(extra["head_dim"])
        scale_divisor = extra["scale_divisor"]

        qkv = F.linear(x, qkv_weight, qkv_bias)

        batch_size = qkv.shape[0]

        q, k, v = qkv.split(d_model, dim=-1)

        q = q.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / scale_divisor

        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        return torch.matmul(attn, v)
    
    
def print_shape_prop(fx_model: fx.GraphModule) -> None:
    print()
    print(
        f"{'Node Name':<45} | "
        f"{'Op':<14} | "
        f"{'Target':<45} | "
        f"{'Shape':<22} | "
        f"{'DType':<14} | "
        f"{'Device'}"
    )
    print("-" * 160)

    for node in fx_model.graph.nodes:
        shape = node.meta.get("shape", "N/A")
        dtype = node.meta.get("dtype", "N/A")
        device = node.meta.get("device", "N/A")

        shape_val = str(shape)
        dtype_val = str(dtype).replace("torch.", "") if dtype != "N/A" else "N/A"
        device_val = str(device)

        target_val = str(node.target)

        if len(target_val) > 45:
            target_val = target_val[:42] + "..."

        print(
            f"{node.name:<45} | "
            f"{node.op:<14} | "
            f"{target_val:<45} | "
            f"{shape_val:<22} | "
            f"{dtype_val:<14} | "
            f"{device_val}"
        )