import torch
import torch.fx as fx
import torch.nn as nn
import flatbuffers as fbs

from pathlib import Path
from typing import Any, Optional, Tuple
import operator

from fxfusion.passes.memory_plan import TensorAlloc
from fxfusion.passes.fusion import Fusion

import gen.fxfusion.DType as DType
import gen.fxfusion.Graph as Graph
import gen.fxfusion.Dim3 as Dim3
import gen.fxfusion.Node as Node
import gen.fxfusion.OpCode as OpCode
import gen.fxfusion.OpAttributes as OpAttributes
import gen.fxfusion.Tensor as Tensor
import gen.fxfusion.TensorKind as TensorKind
import gen.fxfusion.KernelConfig as KernelConfig

import gen.fxfusion.Conv2dAttrs as Conv2dAttrs
import gen.fxfusion.Pool2dAttrs as Pool2dAttrs
import gen.fxfusion.AdaptivePool2dAttrs as AdaptivePool2dAttrs


class Serializer:
    def __init__(self, fx_model: fx.GraphModule) -> None:
        self.fx_model = fx_model
        self.graph = self.fx_model.graph
        self.modules = dict(self.fx_model.named_modules())
        self.builder = fbs.Builder(1024)

        root_dir = Path(__file__).resolve().parents[2]
        self._bin_path = root_dir / "data" / "graph.bin"

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

        Graph.Start(self.builder)
        Graph.AddArenaSize(self.builder, self.fx_model.meta["arena_size"])
        Graph.AddTensors(self.builder, tensors_vec)
        Graph.AddNodes(self.builder, nodes_vec)
        graph_root = Graph.End(self.builder)

        self.builder.Finish(graph_root)

        self._bin_path.parent.mkdir(parents=True, exist_ok=True)
        self._bin_path.write_bytes(bytes(self.builder.Output()))

        return self._bin_path

    def _build_tensor(self, fx_node: fx.Node) -> int:
        tensor_id = self._tensor_id(fx_node)
        alloc: Optional[TensorAlloc] = fx_node.meta.get("alloc")

        name_offset = self.builder.CreateString(fx_node.name)

        shape = list(fx_node.meta.get("shape") or [])
        shape_offset = self._build_int_vector(shape, Tensor.StartShapeVector)

        data_offset = self._build_data_vector(fx_node)

        Tensor.Start(self.builder)
        Tensor.AddId(self.builder, tensor_id)
        Tensor.AddName(self.builder, name_offset)
        Tensor.AddKind(self.builder, self._tensor_kind(fx_node))
        Tensor.AddShape(self.builder, shape_offset)
        Tensor.AddDtype(self.builder, self._dtype_map(fx_node.meta.get("dtype")))

        if alloc is not None:
            Tensor.AddOffset(
                self.builder,
                alloc.mem_offset if alloc.mem_offset is not None else -1,
            )
            Tensor.AddSizeBytes(self.builder, alloc.size_bytes)
            Tensor.AddAliasOf(self.builder, self._alias_of(alloc))

        if data_offset is not None:
            Tensor.AddData(self.builder, data_offset)

        return Tensor.End(self.builder)

    def _build_instruction_node(self, fx_node: fx.Node) -> Optional[int]:
        opcode, attrs_type, attrs_offset = self._node_opcode_and_attrs(fx_node)

        if opcode is None:
            return None

        name_offset = self.builder.CreateString(fx_node.name)

        input_ids = [self._tensor_id(inp) for inp in fx_node.all_input_nodes]
        inputs_offset = self._build_uint_vector(input_ids, Node.StartInputsVector)

        output_ids = [self._tensor_id(fx_node)]
        outputs_offset = self._build_uint_vector(output_ids, Node.StartOutputsVector)

        kernel_config_offset = self._build_kernel_config(fx_node)

        Node.Start(self.builder)
        Node.AddName(self.builder, name_offset)
        Node.AddOpcode(self.builder, opcode)
        Node.AddInputs(self.builder, inputs_offset)
        Node.AddOutputs(self.builder, outputs_offset)

        if attrs_offset is not None:
            Node.AddAttrsType(self.builder, attrs_type)
            Node.AddAttrs(self.builder, attrs_offset)

        if kernel_config_offset is not None:
            Node.AddKernelConfig(self.builder, kernel_config_offset)

        return Node.End(self.builder)

    def _node_opcode_and_attrs(
        self,
        node: fx.Node,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:

        if node.op == "placeholder":
            return OpCode.OpCode.Placeholder, None, None

        if node.op in ("get_attr", "output"):
            return None, None, None

        if node.op == "call_function":
            if node.target in (torch.flatten, torch.reshape):
                return OpCode.OpCode.View, None, None

            if node.target in (torch.add, operator.add):
                return OpCode.OpCode.Add, None, None

            if node.target == Fusion.fused_add_relu:
                return OpCode.OpCode.AddRelu, None, None

            if node.target == Fusion.fused_linear_relu:
                return OpCode.OpCode.LinearRelu, None, None

            if node.target == Fusion.fused_conv2d:
                attrs_offset = self._build_conv2d_attrs(node.meta["attrs"])
                return OpCode.OpCode.Conv2d, OpAttributes.OpAttributes.Conv2dAttrs, attrs_offset

            if node.target == Fusion.fused_conv2d_relu:
                attrs_offset = self._build_conv2d_attrs(node.meta["attrs"])
                return OpCode.OpCode.Conv2dRelu, OpAttributes.OpAttributes.Conv2dAttrs, attrs_offset

            raise RuntimeError(f"Unsupported function: {node.target}")

        if node.op == "call_module":
            mod = self.modules[node.target]

            if isinstance(mod, nn.Conv2d):
                attrs_offset = self._build_conv2d_attrs(mod)
                return OpCode.OpCode.Conv2d, OpAttributes.OpAttributes.Conv2dAttrs, attrs_offset

            if isinstance(mod, nn.MaxPool2d):
                attrs_offset = self._build_pool2d_attrs(mod)
                return OpCode.OpCode.MaxPool2d, OpAttributes.OpAttributes.Pool2dAttrs, attrs_offset

            if isinstance(mod, nn.AvgPool2d):
                attrs_offset = self._build_pool2d_attrs(mod)
                return OpCode.OpCode.AvgPool2d, OpAttributes.OpAttributes.Pool2dAttrs, attrs_offset

            if isinstance(mod, nn.AdaptiveAvgPool2d):
                attrs_offset = self._build_adaptive_pool2d_attrs(mod)
                return (
                    OpCode.OpCode.AdaptiveAvgPool2d,
                    OpAttributes.OpAttributes.AdaptivePool2dAttrs,
                    attrs_offset,
                )

            if isinstance(mod, nn.Linear):
                return OpCode.OpCode.Linear, None, None

            if isinstance(mod, nn.ReLU):
                return OpCode.OpCode.Relu, None, None

            raise RuntimeError(f"Unsupported module {node.target}: {type(mod)}")

        raise RuntimeError(f"Unsupported FX node: {node.op}, {node.target}")

    def _build_conv2d_attrs(self, source: nn.Conv2d | dict[str, Any]) -> int:
        if isinstance(source, nn.Conv2d):
            stride = source.stride
            padding = source.padding
            dilation = source.dilation
            groups = source.groups
        else:
            stride = source["stride"]
            padding = source["padding"]
            dilation = source["dilation"]
            groups = source["groups"]

        stride_offset = self._build_int_vector(self._pair(stride), Conv2dAttrs.StartStrideVector)
        padding_offset = self._build_int_vector(self._pair(padding), Conv2dAttrs.StartPaddingVector)
        dilation_offset = self._build_int_vector(self._pair(dilation), Conv2dAttrs.StartDilationVector)

        Conv2dAttrs.Start(self.builder)
        Conv2dAttrs.AddStride(self.builder, stride_offset)
        Conv2dAttrs.AddPadding(self.builder, padding_offset)
        Conv2dAttrs.AddDilation(self.builder, dilation_offset)
        Conv2dAttrs.AddGroups(self.builder, int(groups))
        return Conv2dAttrs.End(self.builder)

    def _build_pool2d_attrs(self, mod: Any) -> int:
        kernel_size = self._build_int_vector(
            self._pair(mod.kernel_size),
            Pool2dAttrs.StartKernelSizeVector,
        )
        stride = self._build_int_vector(
            self._pair(mod.stride),
            Pool2dAttrs.StartStrideVector,
        )
        padding = self._build_int_vector(
            self._pair(mod.padding),
            Pool2dAttrs.StartPaddingVector,
        )
        dilation = self._build_int_vector(
            self._pair(getattr(mod, "dilation", 1)),
            Pool2dAttrs.StartDilationVector,
        )
        ceil_mode = bool(getattr(mod, "ceil_mode", False))

        Pool2dAttrs.Start(self.builder)
        Pool2dAttrs.AddKernelSize(self.builder, kernel_size)
        Pool2dAttrs.AddStride(self.builder, stride)
        Pool2dAttrs.AddPadding(self.builder, padding)
        Pool2dAttrs.AddDilation(self.builder, dilation)
        Pool2dAttrs.AddCeilMode(self.builder, ceil_mode)
        return Pool2dAttrs.End(self.builder)

    def _build_adaptive_pool2d_attrs(self, mod: nn.AdaptiveAvgPool2d) -> int:
        output_size = self._build_int_vector(
            self._pair(mod.output_size),
            AdaptivePool2dAttrs.StartOutputSizeVector,
        )

        AdaptivePool2dAttrs.Start(self.builder)
        AdaptivePool2dAttrs.AddOutputSize(self.builder, output_size)
        return AdaptivePool2dAttrs.End(self.builder)

    def _build_kernel_config(self, node: fx.Node) -> Optional[int]:
    
            KernelConfig.Start(self.builder)
            
            # Temporary
            KernelConfig.AddGrid(self.builder, Dim3.CreateDim3(self.builder, 1, 1, 1))
            KernelConfig.AddBlock(self.builder, Dim3.CreateDim3(self.builder, 256, 1, 1))
            KernelConfig.AddSharedMemoryBytes(self.builder, 0)
            
            return KernelConfig.End(self.builder)

    def _build_int_vector(self, values: list[int], start_vector_fn) -> int:
        start_vector_fn(self.builder, len(values))
        for value in reversed(values):
            self.builder.PrependInt32(int(value))
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
        target_atoms = target.split(".")
        attr_itr = self.fx_model

        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                )
            attr_itr = getattr(attr_itr, atom)

        return attr_itr

    def _tensor_kind(self, node: fx.Node) -> int:
        kind = node.meta.get("kind", "")

        mapping = {
            "input": TensorKind.TensorKind.Input,
            "const": TensorKind.TensorKind.Constant,
            "activation": TensorKind.TensorKind.Activation,
            "alias": TensorKind.TensorKind.Alias,
            "output": TensorKind.TensorKind.Output,
        }

        if kind not in mapping:
            raise RuntimeError(f"Unsupported tensor kind: {kind}")

        return mapping[kind]

    def _dtype_map(self, dtype: Optional[torch.dtype]) -> int:
        if dtype == torch.float32:
            return DType.DType.Float32
        if dtype == torch.float16:
            return DType.DType.Float16
        if dtype == torch.int32:
            return DType.DType.Int32
        if dtype == torch.int64:
            return DType.DType.Int64

        raise RuntimeError(f"Unsupported dtype: {dtype}")

    def _alias_of(self, alloc: Optional[TensorAlloc]) -> int:
        if alloc is None or alloc.alias_of is None:
            return -1

        for node in self.graph.nodes:
            if node.name == alloc.alias_of:
                return self._tensor_id(node)

        raise RuntimeError(f"Alias target not found: {alloc.alias_of}")

    def _tensor_id(self, node: fx.Node) -> int:
        if "id" not in node.meta:
            raise RuntimeError(
                f"Missing node.meta['id'] for node {node.name}. "
                "Run MemoryPlanningPass before Serializer."
            )

        return int(node.meta["id"])

    def _validate_metadata(self) -> None:
        if "arena_size" not in self.fx_model.meta:
            raise RuntimeError(
                "Missing fx_model.meta['arena_size']. "
                "Run MemoryPlanningPass before Serializer."
            )

        for node in self.graph.nodes:
            for key in ("id", "kind", "alloc"):
                if key not in node.meta:
                    raise RuntimeError(
                        f"Missing node.meta['{key}'] for node {node.name}. "
                        "Run MemoryPlanningPass before Serializer."
                    )

    def _pair(self, value: Any) -> list[int]:
        if isinstance(value, int):
            return [value, value]

        if value is None:
            return [0, 0]

        values = list(value)

        if len(values) == 1:
            return [int(values[0]), int(values[0])]

        return [int(values[0]), int(values[1])]