from typing import Tuple
import torch.fx as fx
from fxfusion.passes.fusion_pass import FusionPass
from fxfusion.passes.shape_prop import ShapePropPass
from fxfusion.serializer import Serializer
from fxfusion.passes.memory_plan import MemoryPlanningPass, print_alloc

class Compiler:
    def __init__(self, DEBUG: bool = False):
        self.DEBUG = DEBUG

    def run(self, model, *example_inputs, model_name: str | None = None)  -> Tuple[fx.GraphModule, str]:
        name = model_name or model.__class__.__name__.lower()

        fx_model = FusionPass().run(model)
        ShapePropPass(fx_model).propagate(*example_inputs)
        MemoryPlanningPass(fx_model).run()

        bin_path = Serializer(fx_model, model_name=name).run()

        if self.DEBUG:
            fx_model.graph.print_tabular()
            print()
            print_alloc(fx_model)

        return fx_model, str(bin_path)