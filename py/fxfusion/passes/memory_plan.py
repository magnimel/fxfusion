import torch
import torch.nn as nn
import torch.fx as fx
from math import prod
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

@dataclass
class TensorAlloc:
    node_name: str
    size_bytes: int
    mem_offset: Optional[int]
    kind: str # "input", "const", "activation", "output", "alias"
    alias_of: Optional[str] = None

@dataclass
class MemoryPlan:
    spec_dict: Dict[str, TensorAlloc] = field(default_factory=dict)
    arena_size: int = 0

class MemoryManager:
    ALIGNMENT = 8
    
    def __init__(self):
        # The manager wholly owns the memory state
        self.free_blocks: List[Tuple[int, int]] = []
        self.arena_top: int = 0
        self.peak_arena_top: int = 0
        

    def _align(self, value: int) -> int:
        return ((value + self.ALIGNMENT - 1) // self.ALIGNMENT) * self.ALIGNMENT
        
    def tensor_size_bytes(self, node: fx.Node) -> int:
        dtype, shape = node.meta.get('dtype'), node.meta.get('shape')
        
        if dtype is None or shape is None:
            raise RuntimeError(f"dtype and shape missing for node {node.name}")
        
        num_elements = prod(shape)
        bytes_per_element = dtype.itemsize
            
        return self._align(num_elements * bytes_per_element)

    def allocate(self, size: int) -> int:
        """Requests an aligned offset for a given size."""
        # 1. Try to find a recycled block (First-Fit)
        
        for idx, (block_offset, block_size) in enumerate(self.free_blocks):
            aligned_offset = self._align(block_offset)
            
            if aligned_offset + size <= block_offset + block_size: 
                self.free_blocks.pop(idx) 

                leftover_front = aligned_offset - block_offset
                if leftover_front > 0:
                    self.free_blocks.append((block_offset, leftover_front))
                    
                leftover_back = (block_size - size) - leftover_front
                if leftover_back > 0:        
                    self.free_blocks.append((aligned_offset + size, leftover_back))
                    
                return aligned_offset
                
        # 2. If no free block fits, bump the arena top
        aligned_offset = self._align(self.arena_top)
        self.arena_top = aligned_offset + size
        self.peak_arena_top = max(self.peak_arena_top, self.arena_top)
        return aligned_offset

    def release(self, offset: int, size: int) -> None: 
        """Marks a node's memory as available and coalesces adjacent blocks."""
        self.free_blocks.append((offset, size))    
        self._coalesce()
        
        
        if self.free_blocks:
            last_block_offset, last_block_size = self.free_blocks[-1]
            
            if last_block_offset + last_block_size == self.arena_top:
                self.free_blocks.pop()
                
                self.arena_top = last_block_offset
        
        
    def _coalesce(self, DEBUG: bool = False) -> None: 
        if not self.free_blocks:
            return
            
        self.free_blocks.sort()
        merged = []
            
        for offset, size in self.free_blocks:
            if not merged:
                merged.append((offset, size))
                continue
                
            prev_offset, prev_size = merged[-1]
            
            if prev_offset + prev_size == offset:
                merged[-1] = (prev_offset, prev_size + size)
                if DEBUG:
                    print(f"Coalesced blocks: ({prev_offset}, {prev_size}) + ({offset}, {size})")
            else:
                merged.append((offset, size))

        self.free_blocks[:] = merged

class MemoryPlanningPass:    
    
    def __init__(self, fx_model: fx.GraphModule) -> None:
        self.fx_model: fx.GraphModule = fx_model
        self.graph: fx.Graph = self.fx_model.graph
        self.memory_manager: MemoryManager = MemoryManager()
    
    def _node_kind(self, node: fx.Node) -> str:
        
        if node.op == "call_function" and node.target in (torch.flatten, torch.reshape):
            return "alias"
        
        if node.op == "call_module":
            mod = dict(self.fx_model.named_modules())[node.target]
            if isinstance(mod, nn.ReLU):
                x = node.args[0]
                if isinstance(x, fx.Node):
                    if x.op == "placeholder":
                        return "activation"
                    if len(x.users) == 1:
                        return "alias"
                return "activation"  
             
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

    def run(self) -> MemoryPlan:
        plan = MemoryPlan()
        nodes = list(self.graph.nodes)
        last_use = self._compute_last_use(nodes)
                    
        for exec_id, node in enumerate(nodes):
            kind = self._node_kind(node)
            node.meta['kind'] = kind
            
            try:
                size = self.memory_manager.tensor_size_bytes(node)
            except RuntimeError:
                size = 0
                
            if kind in ('input', 'const'):
                alloc = TensorAlloc(node.name, size, None, kind)
                
            elif kind == 'activation':
                # Rely on the manager for allocation state
                offset = self.memory_manager.allocate(size)
                alloc = TensorAlloc(node.name, size, offset, kind)
                
                # Check for dead tensors and issue free commands
                dead_args = [
                    arg for arg in node.all_input_nodes 
                    if last_use.get(arg) == exec_id and self._node_kind(arg) == 'activation'
                ]
                
                for arg in dead_args:
                    arg_alloc: TensorAlloc = arg.meta.get('alloc')
                    if arg_alloc is not None and arg_alloc.mem_offset is not None:
                        self.memory_manager.release (
                            arg_alloc.mem_offset, 
                            arg_alloc.size_bytes
                        )
                        
            elif kind in ("alias", "output"):
                returned_node = node.args[0]
                
                if isinstance(returned_node, fx.Node):
                    base_alloc = returned_node.meta["alloc"]
                    alloc = TensorAlloc(
                        node_name=node.name,
                        size_bytes=base_alloc.size_bytes,
                        mem_offset=base_alloc.mem_offset,
                        kind=kind,
                        alias_of=returned_node.name,
                    )
                else:
                    alloc = TensorAlloc(node.name, 0, None, kind)
                
            else:
                raise RuntimeError(f"Unsupported node op: {node.op}")

            node.meta['alloc'] = alloc
            node.meta['id'] = exec_id
            plan.spec_dict[node.name] = alloc

        # The final arena size is whatever the manager dictates
        plan.arena_size = self.memory_manager.peak_arena_top
        self.fx_model.meta["arena_size"] = plan.arena_size
        
        return plan
    
def print_alloc(fx_model: fx.GraphModule):
    print(f"ARENA SIZE: {fx_model.meta["arena_size"]}")
    print(f"{'Node Name':<35} | {'Kind':<12} | {'Size (B)':<10} | {'Offset':<10} | {'Alias Of'}")
    print("-" * 85)

    for node in fx_model.graph.nodes:
        if 'alloc' in node.meta:
            alloc = node.meta['alloc']
            offset_val = str(alloc.mem_offset) if alloc.mem_offset is not None else "N/A"
            alias_of = str(alloc.alias_of) if alloc.alias_of is not None else "N/A"
            print(f"{node.name:<35} | {alloc.kind:<12} | {alloc.size_bytes:<10} | {offset_val:<10} | {alias_of}")
        else:
            print(f"{node.name:<35} | {'Missing Alloc Info'}")