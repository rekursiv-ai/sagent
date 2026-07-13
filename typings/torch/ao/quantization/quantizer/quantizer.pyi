from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node

import torch

__all__ = [
    "DerivedQuantizationSpec",
    "EdgeOrNode",
    "FixedQParamsQuantizationSpec",
    "QuantizationAnnotation",
    "QuantizationSpec",
    "QuantizationSpecBase",
    "Quantizer",
    "SharedQuantizationSpec",
]

class QuantizationSpecBase(ABC): ...

@dataclass(eq=True, frozen=True)
class QuantizationSpec(QuantizationSpecBase):
    dtype: torch.dtype
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor
    quant_min: int | None = ...
    quant_max: int | None = ...
    qscheme: torch.qscheme | None = ...
    ch_axis: int | None = ...
    is_dynamic: bool = ...
    def __post_init__(self) -> None: ...

@dataclass(eq=True, frozen=True)
class FixedQParamsQuantizationSpec(QuantizationSpecBase):
    dtype: torch.dtype
    scale: float
    zero_point: int
    quant_min: int | None = ...
    quant_max: int | None = ...
    qscheme: torch.qscheme | None = ...
    is_dynamic: bool = ...

type EdgeOrNode = tuple[Node, Node] | Node

@dataclass(eq=True, frozen=True)
class SharedQuantizationSpec(QuantizationSpecBase):
    edge_or_node: EdgeOrNode

@dataclass(eq=True, frozen=True)
class DerivedQuantizationSpec(QuantizationSpecBase):
    derived_from: list[EdgeOrNode]
    derive_qparams_fn: Callable[[list[ObserverOrFakeQuantize]], tuple[Tensor, Tensor]]
    dtype: torch.dtype
    quant_min: int | None = ...
    quant_max: int | None = ...
    qscheme: torch.qscheme | None = ...
    ch_axis: int | None = ...
    is_dynamic: bool = ...

@dataclass
class QuantizationAnnotation:
    input_qspec_map: dict[Node, QuantizationSpecBase | None] = ...
    output_qspec: QuantizationSpecBase | None = ...
    allow_implicit_sharing: bool = ...
    _annotated: bool = ...

class Quantizer(ABC):
    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule: ...
    @abstractmethod
    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule: ...
    @abstractmethod
    def validate(self, model: torch.fx.GraphModule) -> None: ...
    def prepare_obs_or_fq_callback(
        self,
        model: torch.fx.GraphModule,
        edge_or_node_to_obs_or_fq: dict[EdgeOrNode, ObserverOrFakeQuantize],
    ) -> None: ...
