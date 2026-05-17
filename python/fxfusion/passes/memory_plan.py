import torch
import torch.fx as fx
from math import prod
from torch.fx.node import map_arg
from dataclasses import dataclass, field
from typing import Type, Dict, Any, Tuple, Optional, List, cast


@dataclass
class TensorAlloc:
    node_name: str
    size_bytes: int
    mem_offset: Optional[int]
    kind: str # "input", "const", "activation", "output"
    alias_of: Optional[str] = None

@dataclass
class MemoryPlan:
    spec_dict: Dict[str, TensorAlloc] = field(default_factory=dict)
    arena_size: int = 0

class MemoryPlanPass():    
    
    ALIGNMENT = 8
    
    def __init__(self, fx_model: fx.GraphModule) -> None:
        self.fx_model : fx.GraphModule = fx_model
        self.graph : fx.Graph = self.fx_model.graph
    
    def _tensor_size_bytes(self, node: fx.Node):
        dtype, shape = node.meta.get('dtype'), node.meta.get('shape')
        
        if dtype is None or shape is None:
            raise RuntimeError(f"dtype and shape missing for node {node.name}")
        
        num_elements = prod(shape)
        bytes_per_element = dtype.itemsize
            
        return self._align(num_elements * bytes_per_element)


    ## first fit
    def _find_free_block(
        self, 
        size: int,
        free_blocks: List[Tuple[int, int]],
    ) -> Optional[int]: 
        
        for idx, (block_offset, block_size) in enumerate(free_blocks):
            
            aligned_offset = self._align(block_offset)
            
            if aligned_offset + size > block_offset + block_size: 
                continue
            
            free_blocks.pop(idx) 

            leftover_front = aligned_offset - block_offset
            if leftover_front > 0:
                free_blocks.append((block_offset, leftover_front))
                
            leftover_back = (block_size - size) - (aligned_offset - block_offset) 
            if leftover_back > 0:        
                free_blocks.append((aligned_offset + size, leftover_back))
                
            return aligned_offset
                
        return None
    
    
    def _reclaim_block(self, node: fx.Node, free_blocks: List[Tuple[int, int]]) -> bool: 
        alloc: TensorAlloc = node.meta['alloc']
        if alloc.mem_offset is None:
            return False
        
        free_blocks.append((alloc.mem_offset, alloc.size_bytes))    
        return True
    
    def _coalesce(self, free_blocks: List[Tuple[int, int]]) -> None: 
        
        free_blocks.sort()
        merged = []
            
        for offset, size in free_blocks:
            if not merged:
                merged.append((offset, size))
                continue
                
            prev_offset, prev_size = merged[-1]
            
            if prev_offset + prev_size == offset:
                merged[-1] = (prev_offset, prev_size + size)
                
            else:
                merged.append((offset, size))

        free_blocks[:] = merged
        
        return 
        
    def _align(self, value: int, alignment: int = 8) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    def _node_kind(self, node: fx.Node) -> str:
        if node.op == "placeholder":
            return "input"
        if node.op == "get_attr":
            return "const"
        if node.op in ("call_function", "call_method", "call_module"):
            return "activation"
        if node.op == "output":
            return "output"
        raise RuntimeError(f"Unsupported node op: {node.op}")
    

    def _allocate_new_block(self, size: int, arena_top: int):
        aligned_offset = self._align(arena_top, self.ALIGNMENT)
        arena_top = aligned_offset + size
        return aligned_offset, arena_top    
        
    def _compute_last_use(self, nodes: List[fx.Node]) -> Dict[fx.Node, int]:
        last_use: Dict[fx.Node, int] = {}

        for i, node in enumerate(nodes):
            for input_node in node.all_input_nodes:
                last_use[input_node] = i

        return last_use

    def run(self):
        plan: MemoryPlan = MemoryPlan()
        free_blocks: List[Tuple[int, int]] = []
        
        last_use = self._compute_last_use(list(self.graph.nodes))
        
        arena_top = 0
                    
        for exec_id, node in enumerate(self.graph.nodes):
            
            kind = self._node_kind(node)
            
            try:
                size = self._tensor_size_bytes(node)
            except RuntimeError:
                size = 0
                
            if kind in ('input', 'const'):
                alloc = TensorAlloc(node.name, size, None, kind)
                
            elif kind == 'activation':
                
                offset = self._find_free_block(size, free_blocks)
                if offset is None:
                    offset, arena_top = self._allocate_new_block(size, arena_top)
                
                alloc = TensorAlloc(node.name, size, offset, kind)
                
                dead_args = [
                    arg for arg in node.all_input_nodes 
                    if last_use.get(arg) == exec_id and self._node_kind(arg) == 'activation'
                ]
                
                if dead_args:
                    for arg in dead_args:
                        self._reclaim_block(arg, free_blocks)
                    self._coalesce(free_blocks)
                        
            elif kind == 'output':
                    returned_node = node.args[0]

                    if isinstance(returned_node, fx.Node):
                        base_alloc = returned_node.meta["alloc"]

                        alloc = TensorAlloc(
                            node_name=node.name,
                            size_bytes=base_alloc.size_bytes,
                            mem_offset=base_alloc.mem_offset,
                            kind="alias",
                            alias_of=returned_node.name,
                        )
                
            else:
                raise RuntimeError(f"Unsupported node op: {node.op}")

            node.meta['alloc'] = alloc
            plan.spec_dict[node.name] = alloc
            plan.arena_size = arena_top
            
        return plan
    
    def print_alloc(self):
        
        print(f"{'Node Name':<28} | {'Kind':<12} | {'Size (Bytes)':<15} | {'Offset'}")
        print("-" * 75)

        for node in self.graph.nodes:
            if 'alloc' in node.meta:
                alloc = node.meta['alloc']
                
                offset_val = str(alloc.mem_offset) if alloc.mem_offset is not None else "N/A"
                
                print(f"{node.name:<28} | {alloc.kind:<12} | {alloc.size_bytes:<15} | {offset_val}")
            else:
                print(f"{node.name:<28} | {'Missing Alloc Info'}")