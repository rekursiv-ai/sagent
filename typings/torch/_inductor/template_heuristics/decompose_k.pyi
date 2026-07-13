import torch

from .base import TemplateConfigHeuristics
from .gemm import GemmMaxAutotuneTemplateConfigHeuristics
from .registry import register_template_heuristic
from ..kernel.mm import decompose_k_subgraph_template

@register_template_heuristic(decompose_k_subgraph_template.uid, None, op_name="mm")
class EmptyDecomposeKConfigHeuristics(TemplateConfigHeuristics): ...

@register_template_heuristic(
    decompose_k_subgraph_template.uid,
    "cuda",
    register=torch.version.hip is None,
    op_name="mm",
)
class DecomposeKConfigHeuristics(GemmMaxAutotuneTemplateConfigHeuristics): ...
