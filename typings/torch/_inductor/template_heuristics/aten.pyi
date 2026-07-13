from typing import Any

from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic
from ..ir import Layout
from ..kernel.bmm import aten_baddbmm, aten_bmm, aten_bmm_dtype
from ..kernel.mm import aten__fp8_mm, aten__int_mm, aten_addmm, aten_bias_addmm, aten_mm
from ..kernel.mm_plus_mm import aten_mm_plus_mm
from ..kernel_inputs import KernelInputs

@register_template_heuristic(aten_mm.uid, None)
@register_template_heuristic(aten__fp8_mm.uid, None)
@register_template_heuristic(aten__int_mm.uid, None)
@register_template_heuristic(aten_bmm.uid, None)
@register_template_heuristic(aten_mm_plus_mm.uid, None)
@register_template_heuristic(aten_bmm_dtype.uid, "cuda")
class ATenConfigHeuristics(TemplateConfigHeuristics): ...

@register_template_heuristic(aten_addmm.uid, None, op_name="addmm")
@register_template_heuristic(aten_baddbmm.uid, None, op_name="baddbmm")
class ATenAddMMConfigHeuristics(ATenConfigHeuristics):
    def get_extra_kwargs(
        self, kernel_inputs: KernelInputs, layout: Layout, op_name: str
    ) -> dict[str, Any]: ...

@register_template_heuristic(aten_bias_addmm.uid, None, op_name="addmm")
class ATenBiasAddMMConfigHeuristics(
    ATenAddMMConfigHeuristics, GemmMaxAutotuneTemplateConfigHeuristics
): ...
