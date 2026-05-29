import torch
import torch.nn as nn
import fxfusion.lib.fxfusion_extension
from typing import List, Optional
from fxfusion.compiler import Compiler


class Engine:
    def __init__(
        self, 
        model: nn.Module, 
        example_inputs: List[torch.Tensor], 
        model_name: Optional[str] = None,
        device: str = "cpu",
        DEBUG: bool = True,
    ):
        self.compiler = Compiler(DEBUG=DEBUG)

        self.fx_model, self.bin_path = self.compiler.run(
            model,
            *example_inputs,
            model_name=model_name,
        )

      
        self._engine = torch.classes.fxfusion_extension.ExecutionEngine(
            self.bin_path,
            device,
        )

    def run(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return self._engine.run(inputs)