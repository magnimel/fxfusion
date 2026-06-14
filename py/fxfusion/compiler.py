from typing import Tuple

import torch
import torch.nn as nn
import torch.fx as fx
from fxfusion.passes.fusion_pass import FusionPass
from fxfusion.passes.shape_prop import ShapePropPass, print_shape_prop
from fxfusion.passes.memory_plan import MemoryPlanningPass, print_alloc
from fxfusion.serializer import Serializer


class Compiler:
    def __init__(self, DEBUG: bool = False, memory_alignment: int = 64):
        self.DEBUG = DEBUG
        self.memory_alignment = memory_alignment

    def run(
        self, model: nn.Module, *example_inputs, model_name: str | None = None
    ) -> Tuple[fx.GraphModule, str]:
        
        if model.training:
            raise RuntimeError(
                "FXFusion only supports inference graphs. "
                "Call model.eval() before compiling."
            )

        name = model_name or model.__class__.__name__.lower()

        with torch.inference_mode():
            
            fx_model = FusionPass().run(model)
            if self.DEBUG:
                fx_model.graph.print_tabular()

            ShapePropPass(fx_model).propagate(*example_inputs)
            if self.DEBUG:
                print_shape_prop(fx_model)

            MemoryPlanningPass(fx_model, alignment=self.memory_alignment).run()
            if self.DEBUG:
                print_alloc(fx_model)

            bin_path = Serializer(fx_model, model_name=name).run()

        return fx_model, str(bin_path)
