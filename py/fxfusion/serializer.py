from platform import node

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import flatbuffers as fbs

from pathlib import Path
from typing import Any, Optional, Tuple
import operator

from fxfusion.passes.memory_plan import TensorAlloc
from fxfusion.passes.fusion.symbols import Fusion

import gen.fxfusion.DType as DType
import gen.fxfusion.Graph as Graph
import gen.fxfusion.Node as Node
import gen.fxfusion.OpCode as OpCode
import gen.fxfusion.Tensor as Tensor
import gen.fxfusion.TensorKind as TensorKind


class Serializer:
    def __init__(self, fx_model: fx.GraphModule, model_name: str = "model") -> None:
        self.fx_model = fx_model
        self.model_name = model_name
        self.graph = self.fx_model.graph
        self.modules = dict(self.fx_model.named_modules())
        self.builder = fbs.Builder(1024)
        root_dir = Path(__file__).resolve().parents[2]
        self._bin_path = root_dir / "data" / f"{model_name}.bin"

    def run(self) -> Path:
        self._validate_metadata()

        tensor_offsets = [self._build_tensor(node) for node in self.graph.nodes]

        Graph.StartTensorsVector(self.builder, len(tensor_offsets))
        for offset in reversed(tensor_offsets):
            self.builder.PrependUOffsetTRelative(offset)
        tensors_vec = self.builder.EndVector()

        instruction_offsets = []
        for node in self.graph.nodes:
            node_offset = self._build_instruction_node(node)
            if node_offset is not None:
                instruction_offsets.append(node_offset)

        Graph.StartNodesVector(self.builder, len(instruction_offsets))
        for offset in reversed(instruction_offsets):
            self.builder.PrependUOffsetTRelative(offset)
        nodes_vec = self.builder.EndVector()

        name_offset = self.builder.CreateString(self.model_name)

        Graph.Start(self.builder)
        Graph.AddName(self.builder, name_offset)
        Graph.AddArenaSize(self.builder, self.fx_model.meta["arena_size"])
        Graph.AddTensors(self.builder, tensors_vec)
        Graph.AddNodes(self.builder, nodes_vec)
        graph_root = Graph.End(self.builder)

        self.builder.Finish(graph_root)

        self._bin_path.parent.mkdir(parents=True, exist_ok=True)
        self._bin_path.write_bytes(bytes(self.builder.Output()))

        return self._bin_path

    def _input_ids_for_node(self, node: fx.Node) -> list[int]:
        input_nodes: list[fx.Node] = []

        def collect_input(arg: fx.Node) -> fx.Node:
            input_nodes.append(arg)
            return arg
    
        fx.map_arg(node.args, collect_input)
        fx.map_arg(node.kwargs, collect_input)

        return [self._tensor_id(input_node) for input_node in input_nodes]

    def _build_instruction_node(self, fx_node: fx.Node) -> Optional[int]:
        result = self._node_opcode_and_params(fx_node)
        if result is None:
            return None

        opcode, int_params, float_params = result

        input_ids = self._input_ids_for_node(fx_node)
        inputs_offset = self._build_uint_vector(input_ids, Node.StartInputIdsVector)

        output_ids = [self._tensor_id(fx_node)]
        outputs_offset = self._build_uint_vector(output_ids, Node.StartOutputIdsVector)

        int_params_offset = self._build_int_vector(
            int_params,
            Node.StartIntParamsVector,
        )

        float_params_offset = self._build_float_vector(
            float_params,
            Node.StartFloatParamsVector,
        )

        Node.Start(self.builder)
        Node.AddOpCode(self.builder, opcode)
        Node.AddInputIds(self.builder, inputs_offset)
        Node.AddOutputIds(self.builder, outputs_offset)
        Node.AddIntParams(self.builder, int_params_offset)
        Node.AddFloatParams(self.builder, float_params_offset)
        return Node.End(self.builder)

    def _node_opcode_and_params(self, node: fx.Node) -> Optional[Tuple[int, list[int], list[float]]]:

        if node.op == "placeholder":
            return OpCode.OpCode.NoOp, [], []
        if node.op in ("get_attr", "output"):
            return OpCode.OpCode.NoOp, [], []
        
        if node.op == "call_function":
            if node.target in (torch.flatten, torch.reshape):
                return OpCode.OpCode.NoOp, [], []
            if node.target == torch.narrow:
                return OpCode.OpCode.Narrow, *self._narrow_params(node)
            if node.target == torch.transpose:
                return OpCode.OpCode.Transpose, *self._transpose_params(node)
            if node.target in (torch.add, operator.add):
                return OpCode.OpCode.Add, [], []
            if node.target in (torch.mul, operator.mul):
                return OpCode.OpCode.Mul, *self._mul_params(node)
            if node.target == F.dropout:
                return OpCode.OpCode.NoOp, [], []
            if node.target == Fusion.relu:
                return OpCode.OpCode.Relu, [], []
            if node.target == Fusion.add_relu:
                return OpCode.OpCode.AddRelu, [], []
            if node.target == Fusion.linear:
                return OpCode.OpCode.Linear, [], []
            if node.target == Fusion.linear_relu:
                return OpCode.OpCode.LinearRelu, [], []
            if node.target == Fusion.conv2d:
                return OpCode.OpCode.Conv2d, *self._conv2d_params(node.meta["attrs"])
            if node.target == Fusion.conv2d_relu:
                return OpCode.OpCode.Conv2dRelu, *self._conv2d_params(node.meta["attrs"])
            if node.target == Fusion.embedding:
                return OpCode.OpCode.Embedding, [], []
            if node.target == Fusion.layernorm:
                return OpCode.OpCode.LayerNorm, *self._layernorm_params(node)
            if node.target == Fusion.add_layernorm:
                return OpCode.OpCode.AddLayerNorm, *self._layernorm_params(node)
            if node.target == Fusion.mha:
                return OpCode.OpCode.MHA, *self._mha_params(node)
            if node.target == Fusion.feedforward:
                return OpCode.OpCode.FeedForward, [], []

            raise RuntimeError(f"Unsupported function: {node.target}")

        if node.op == "call_method":
            if node.target in ("view", "reshape", "flatten", "contiguous"):
                return OpCode.OpCode.NoOp, [], []
            if node.target == torch.narrow:
                return OpCode.OpCode.Narrow, *self._narrow_params(node)
            if node.target == "transpose":
                return OpCode.OpCode.Transpose, *self._transpose_params(node)
            if node.target == "size":
                return OpCode.OpCode.Size, *self._size_params(node)

            raise RuntimeError(f"Unsupported method: {node.target}")

        if node.op == "call_module":
            mod = self.modules[node.target]

            if isinstance(mod, nn.Dropout):
                return OpCode.OpCode.NoOp, [], []
            if isinstance(mod, nn.MaxPool2d):
                return OpCode.OpCode.MaxPool2d, *self._max_pool2d_params(mod)
            if isinstance(mod, nn.AvgPool2d):
                return OpCode.OpCode.AvgPool2d, *self._avg_pool2d_params(mod)
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                return OpCode.OpCode.AdaptiveAvgPool2d, *self._adaptive_avg_pool2d_params(mod)

            raise RuntimeError(f"Unsupported module {node.target}: {type(mod)}")

        raise RuntimeError(f"Unsupported FX node: {node.op}, {node.target}")

    # grid/block placeholder — will be computed properly for CUDA later
    def _grid_block(self) -> list[int]:
        return [1, 1, 1, 256, 1, 1]

    def _conv2d_params(self, source: nn.Conv2d | dict[str, Any]) -> tuple[list[int], list[float]]:
        if isinstance(source, nn.Conv2d):
            stride   = self._pair(source.stride)
            padding  = self._pair(source.padding)
            dilation = self._pair(source.dilation)
            groups   = int(source.groups)
        else:
            stride   = self._pair(source["stride"])
            padding  = self._pair(source["padding"])
            dilation = self._pair(source["dilation"])
            groups   = int(source["groups"])

        return [*stride, *padding, *dilation, groups, *self._grid_block()], []

    def _max_pool2d_params(self, mod: nn.MaxPool2d) -> tuple[list[int], list[float]]:
        return [
            *self._pair(mod.kernel_size),
            *self._pair(mod.stride),
            *self._pair(mod.padding),
            *self._pair(getattr(mod, "dilation", 1)),
            int(getattr(mod, "ceil_mode", False)),
            *self._grid_block(),
        ], []

    def _avg_pool2d_params(self, mod: nn.AvgPool2d) -> tuple[list[int], list[float]]:
        return [
            *self._pair(mod.kernel_size),
            *self._pair(mod.stride),
            *self._pair(mod.padding),
            int(getattr(mod, "ceil_mode", False)),
            *self._grid_block(),
        ], []

    def _adaptive_avg_pool2d_params(self, mod: nn.AdaptiveAvgPool2d) -> tuple[list[int], list[float]]:
        return [*self._pair(mod.output_size), *self._grid_block()], []

    def _transpose_params(self, node: fx.Node) -> tuple[list[int], list[float]]:
        return [int(node.args[1]), int(node.args[2])], []

    def _size_params(self, node: fx.Node) -> tuple[list[int], list[float]]:
        return [int(node.args[1])], []
    
    def _narrow_params(self, node: fx.Node) -> tuple[list[int], list[float]]:
            dim, start = int(node.args[1]), int(node.args[2])
            return [dim, start], []

    def _mul_params(self, node: fx.Node) -> tuple[list[int], list[float]]:
        lhs, rhs = node.args[0], node.args[1]
        scalar = None
        if isinstance(rhs, (bool, int, float)):
            scalar = rhs
        elif isinstance(lhs, (bool, int, float)):
            scalar = lhs
        if scalar is None:
            return [0], []
        if isinstance(scalar, bool):
            return [3], [1.0 if scalar else 0.0]
        if isinstance(scalar, int):
            return [2], [float(scalar)]
        return [1], [float(scalar)]

    def _layernorm_params(self, node: fx.Node) -> tuple[list[int], list[float]]:
        extra            = node.args[-1]
        normalized_shape = extra["normalized_shape"]
        eps              = extra["eps"]
        return [len(normalized_shape), *normalized_shape], [float(eps)]

    def _mha_params(self, node: fx.Node) -> tuple[list[int], list[float]]:
        extra = node.args[-1]
        return (
            [int(extra["num_heads"]), int(extra["head_dim"]), int(extra["d_model"]), int(extra["qkv_dim"])],
            [float(extra["scale_divisor"])],
        )
        
    def _build_tensor(self, fx_node: fx.Node) -> int:
        tensor_id = self._tensor_id(fx_node)
        alloc: Optional[TensorAlloc] = fx_node.meta.get("alloc")

        shape_offset = self._build_int_vector(list(fx_node.meta.get("shape") or []), Tensor.StartShapeVector)
        data_offset = self._build_data_vector(fx_node)

        Tensor.Start(self.builder)
        Tensor.AddId(self.builder, tensor_id)
        Tensor.AddKind(self.builder, self._tensor_kind(fx_node))
        Tensor.AddShape(self.builder, shape_offset)
        Tensor.AddDtype(self.builder, self._dtype_map(fx_node.meta.get("dtype")))

        if alloc is not None:
            Tensor.AddOffset(self.builder, alloc.mem_offset if alloc.mem_offset is not None else -1)
            Tensor.AddSizeBytes(self.builder, alloc.size_bytes)
            Tensor.AddAliasOf(self.builder, self._alias_of(alloc))

        if data_offset is not None:
            Tensor.AddData(self.builder, data_offset)

        return Tensor.End(self.builder)

    def _build_int_vector(self, values: list[int], start_vector_fn) -> int:
        start_vector_fn(self.builder, len(values))
        for value in reversed(values):
            self.builder.PrependInt32(int(value))
        return self.builder.EndVector()

    def _build_float_vector(self, values: list[float], start_vector_fn) -> int:
        start_vector_fn(self.builder, len(values))
        for value in reversed(values):
            self.builder.PrependFloat32(float(value))
        return self.builder.EndVector()

    def _build_uint_vector(self, values: list[int], start_vector_fn) -> int:
        start_vector_fn(self.builder, len(values))
        for value in reversed(values):
            self.builder.PrependUint32(int(value))
        return self.builder.EndVector()

    def _build_data_vector(self, node: fx.Node) -> Optional[int]:
        if node.meta.get("kind") != "const":
            return None
        value = self._fetch_attr(str(node.target))
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unsupported const type for {node.name}: {type(value)}")
        tensor_bytes = value.detach().contiguous().cpu().numpy().tobytes()
        return self.builder.CreateByteVector(tensor_bytes)

    def _fetch_attr(self, target: str) -> Any:
        attr_itr = self.fx_model
        for i, atom in enumerate(target.split(".")):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {target}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def _tensor_kind(self, node: fx.Node) -> int:
        mapping = {
            "input":      TensorKind.TensorKind.Input,
            "const":      TensorKind.TensorKind.Constant,
            "activation": TensorKind.TensorKind.Activation,
            "alias":      TensorKind.TensorKind.Alias,
            "output":     TensorKind.TensorKind.Output,
        }
        kind = node.meta.get("kind", "")
        if kind not in mapping:
            raise RuntimeError(f"Unsupported tensor kind: {kind}")
        return mapping[kind]

    def _dtype_map(self, dtype: Optional[torch.dtype]) -> int:
        mapping = {
            torch.float32: DType.DType.Float32,
            torch.float16: DType.DType.Float16,
            torch.int32:   DType.DType.Int32,
            torch.int64:   DType.DType.Int64,
            torch.bool:    DType.DType.Bool,
        }
        if dtype not in mapping:
            raise RuntimeError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    def _alias_of(self, alloc: Optional[TensorAlloc]) -> int:
        if alloc is None or alloc.alias_of is None:
            return -1
        for node in self.graph.nodes:
            if node.name == alloc.alias_of:
                return self._tensor_id(node)
        raise RuntimeError(f"Alias target not found: {alloc.alias_of}")

    def _tensor_id(self, node: fx.Node) -> int:
        if "id" not in node.meta:
            raise RuntimeError(f"Missing node.meta['id'] for node {node.name}. Run MemoryPlanningPass before Serializer.")
        return int(node.meta["id"])

    def _validate_metadata(self) -> None:
        if "arena_size" not in self.fx_model.meta:
            raise RuntimeError("Missing fx_model.meta['arena_size']. Run MemoryPlanningPass before Serializer.")
        for node in self.graph.nodes:
            for key in ("id", "kind", "alloc"):
                if key not in node.meta:
                    raise RuntimeError(f"Missing node.meta['{key}'] for node {node.name}. Run MemoryPlanningPass before Serializer.")

    def _pair(self, value: Any) -> list[int]:
        if isinstance(value, int):
            return [value, value]
        if value is None:
            return [0, 0]
        values = list(value)
        if len(values) == 1:
            return [int(values[0]), int(values[0])]
        return [int(values[0]), int(values[1])]