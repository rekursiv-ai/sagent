from collections.abc import Mapping, Sequence
from typing import Any

import abc
import dataclasses

from torch import _prims_common
from torch.onnx._internal.fx import _pass
from torch.utils import _python_dispatch

import torch
import torch._ops
import torch.fx

logger = ...

@dataclasses.dataclass
class TypePromotionSnapshot:
    args_dtypes: Mapping[int, torch.dtype]
    kwargs_dtypes: Mapping[str, torch.dtype]
    out_dtype: torch.dtype

class TypePromotionRule(abc.ABC):
    def __init__(self, namespace: str, op_name: str) -> None: ...
    @abc.abstractmethod
    def __hash__(self) -> int: ...
    @abc.abstractmethod
    def __repr__(self) -> None: ...
    @abc.abstractmethod
    def __eq__(self, other: object) -> bool: ...
    def is_valid(self) -> bool: ...
    @abc.abstractmethod
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot: ...

class ElementwiseTypePromotionRule(TypePromotionRule):
    _USE_OPMATH: bool = ...
    def __init__(
        self,
        namespace: str,
        op_name: str,
        promote_args_positions: Sequence[int],
        promote_kwargs_names: Sequence[str],
        promotion_kind: _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND,
    ) -> None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot: ...

class DivElementwiseTypePromotionRule(ElementwiseTypePromotionRule):
    def __init__(self) -> None: ...
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot: ...

class ReductionTypePromotionRule(TypePromotionRule):
    def __init__(
        self,
        namespace: str,
        op_name: str,
        promotion_kind: _prims_common.REDUCTION_OUTPUT_TYPE_KIND,
    ) -> None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot: ...

class AllOrAnyReductionTypePromotionRule(ReductionTypePromotionRule):
    def __init__(self, op_name: str) -> None: ...
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot: ...

class SumLikeReductionTypePromotionRule(ReductionTypePromotionRule):
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot: ...

_GENERATED_ATEN_TYPE_PROMOTION_RULE_SET = ...
_EXTRA_TYPE_PROMOTION_RULE_SET = ...

class ElementwiseTypePromotionRuleSetGenerator:
    @classmethod
    def generate_from_torch_refs(cls) -> set[ElementwiseTypePromotionRule]: ...

class TypePromotionTable:
    def __init__(self) -> None: ...
    def add_rule(self, rule: TypePromotionRule) -> None: ...
    def get_rule(
        self, py_op: torch._ops.OpOverloadPacket
    ) -> TypePromotionRule | None: ...

def get_type_promotion_rule(
    node: torch.fx.Node, type_promotion_table: TypePromotionTable
) -> TypePromotionRule | None: ...

class _OpTraceDispatchMode(_python_dispatch.TorchDispatchMode):
    def __init__(self, *args, **kwargs) -> None: ...
    def __torch_dispatch__(self, func, types, args=..., kwargs=...): ...

def find_compatible_op_overload(
    op: torch._ops.OpOverloadPacket, args: tuple, kwargs: dict
) -> torch._ops.OpOverload: ...

class _TypePromotionInterpreter(torch.fx.Interpreter):
    def __init__(
        self, module: torch.fx.GraphModule, type_promotion_table: TypePromotionTable
    ) -> None: ...
    def run_node(self, n: torch.fx.Node) -> Any: ...

class InsertTypePromotion(_pass.Transform):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        type_promotion_table: TypePromotionTable | None = ...,
    ) -> None: ...
