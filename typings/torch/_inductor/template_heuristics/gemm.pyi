from .base import TemplateConfigHeuristics
from ..ir import Layout
from ..kernel_inputs import KernelInputs

class GemmMaxAutotuneTemplateConfigHeuristics(TemplateConfigHeuristics):
    def should_run(self, inputs: KernelInputs, layout: Layout) -> bool: ...
