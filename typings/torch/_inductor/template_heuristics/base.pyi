from collections.abc import Generator
from typing import Any

from ..ir import Layout
from ..kernel_inputs import KernelInputs

class TemplateConfigHeuristics:
    def should_run(self, inputs: KernelInputs, layout: Layout) -> bool: ...
    def get_template_configs(
        self, kernel_inputs: KernelInputs, layout: Layout, op_name: str
    ) -> Generator[dict[str, Any]]: ...
    def get_extra_kwargs(
        self, kernel_inputs: KernelInputs, layout: Layout, op_name: str
    ) -> dict[str, Any]: ...
    def adjust_kernel_inputs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> KernelInputs: ...
