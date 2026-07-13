from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import Pattern
from torch.fx.graph import Node

from .quantize_handler import QuantizeHandler

__all__: list[str] = ...
type _MatchResult = tuple[Node, list[Node], Pattern | None, QuantizeHandler]
type _MatchResultWithQConfig = tuple[
    Node, list[Node], Pattern | None, QuantizeHandler, QConfigAny
]
