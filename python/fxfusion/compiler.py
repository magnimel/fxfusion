from fxfusion.passes.fusion_pass import FusionPass
from fxfusion.passes.shape_prop import ShapePropPass

class FXFusionCompiler:
    def __init__(self):
        self.fusion_pass = FusionPass()

    def run(self, model, example_inputs):
        fx_model = self.fusion_pass.run(model)
        ShapePropPass(fx_model).propagate(*example_inputs)
        return fx_model