import torch

from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic
from ..kernel.mm import (
    addmm_contiguous_subgraph_template,
    mm_contiguous_subgraph_template,
)

@register_template_heuristic(mm_contiguous_subgraph_template.uid, None, op_name="mm")
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid, None, op_name="addmm"
)
class EmptyContiguousMMConfigHeuristics(TemplateConfigHeuristics): ...

@register_template_heuristic(
    mm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="mm",
)
@register_template_heuristic(
    addmm_contiguous_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is not None,
    op_name="addmm",
)
class ContiguousMMHeuristics(GemmMaxAutotuneTemplateConfigHeuristics): ...
