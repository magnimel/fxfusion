from fxfusion.passes.fusion_pass import FusionPass
from fxfusion.passes.shape_prop import ShapePropPass
from fxfusion.serializer import Serializer
from fxfusion.passes.memory_plan import MemoryPlanningPass, print_alloc

class Compiler:
    def __init__(self, DEBUG: bool = True):
        self.DEBUG = DEBUG

    def run(self, model, *example_inputs):
        fx_model = FusionPass().run(model)
        
        if self.DEBUG:
            fx_model.graph.print_tabular()
            print()
            
        ShapePropPass(fx_model).propagate(*example_inputs)
        MemoryPlanningPass(fx_model).run()
        
        if self.DEBUG:
            print_alloc(fx_model)
        
        Serializer(fx_model).run()
        
        return fx_model