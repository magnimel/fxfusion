from fxfusion.passes.fusion_pass import FusionPass
from fxfusion.passes.shape_prop import ShapePropPass
from fxfusion.passes.memory_plan import MemoryPlanPass

class FXFusionCompiler:
    def __init__(self):
        self.fusion_pass = FusionPass()
        self.plan = None

    def run(self, model, *example_inputs):
        fx_model = self.fusion_pass.run(model)
        ShapePropPass(fx_model).propagate(*example_inputs)
        memory_pass = MemoryPlanPass(fx_model)
        self.plan = memory_pass.run()
        memory_pass.print_alloc()
        return fx_model