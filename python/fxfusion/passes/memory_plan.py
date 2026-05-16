import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.functional as F
from typing import Type, Dict, Any, Tuple, Optional, List, cast
from fxfusion.passes.fusion import Fusion
from torch.fx.node import map_arg
from dataclasses import dataclass


@dataclass
class SpecAlloc:
    mem_id: int
    mem_obj: str
    mem_offset: int = -1

@dataclass
class MemoryPlan:
    spec_dict: Dict[int, SpecAlloc] = {}
    bufsizes: List[int] = []
    

class MemoryPlanningPass():    
    
    def __init__(self, fx_model: fx.GraphModule) -> None:
        self.memory_plan: MemoryPlan = MemoryPlan()
        self.fx_model : fx.GraphModule = fx_model
        self.graph : fx.Graph = self.fx_model.graph
        self.modules: Dict[str, Any] = dict(self.fx_model.named_modules())
    
    def _get_bufsize(self, node: fx.Node):
        if 'dtype' not in node.meta or 'shape' not in node.meta:
            raise RuntimeWarning("dtype and shape missing for node")
        dummy = torch.zeros(size=node.meta['shape'], dtype = node.meta['dtype'])
        return dummy.untyped_storage().nbytes()

    def _allocate_buf(self, node: fx.Node):
        bufsize = self._get_bufsize(node)
        self.memory_plan.bufsizes.append(bufsize)
        return bufsize

    def run(self):
        
        def _get_offset(i: int):
            return self.memory_plan.bufsizes[i-1] +  \
                self.memory_plan.spec_dict[i-1].mem_offset
    
        ops : set[str] = {'get_attr', 'call_method', 'call_function', 'call_module', 'output'}
        
        for i, node in enumerate(self.graph.nodes):
            result: SpecAlloc = SpecAlloc()
            result.mem_id, result.mem_obj = i, node.name
            self._allocate_buf(node)
            
            self.memory_plan.spec_dict[i] = result
            if node.op == 'placeholder':
                result.mem_offset = 0
            elif node.op in ops:
                result.mem_offset = _get_offset(i)
            else:
                raise RuntimeError(f"Unsupported node op: {node.op}")

            node.meta['spec'] = result
            self.memory_plan.spec_dict[i] = result
            
        return self.memory_plan