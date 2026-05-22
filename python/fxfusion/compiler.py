from fxfusion.passes.fusion_pass import FusionPass
from fxfusion.passes.shape_prop import ShapePropPass
from fxfusion.passes.memory_plan import MemoryPlanningPass

class FXFusionCompiler:
    def __init__(self, DEBUG: bool = True):
        self.fusion_pass = FusionPass()
        self.memory_plan = None
        self.DEBUG = DEBUG

    def run(self, model, *example_inputs):
        fx_model = self.fusion_pass.run(model)
        
        if self.DEBUG:
            fx_model.graph.print_tabular()
            print()
            
        ShapePropPass(fx_model).propagate(*example_inputs)
        memory_pass = MemoryPlanningPass(fx_model)
        self.memory_plan = memory_pass.run()
        
        if self.DEBUG:
            memory_pass.print_alloc()
        
        return fx_model