from collections.abc import Generator
from functools import partial
from typing import Any

import typing

from triton import Config as TritonConfig

import sympy
import torch

from .codegen.common import KernelTemplate
from .codegen.simd_kernel_features import SIMDKernelFeatures
from .codegen.triton import TritonKernel
from .ir import ChoiceCaller
from .kernel_inputs import KernelInputs
from .scheduler import BaseSchedulerNode, Scheduler
from .select_algorithm import ExternKernelChoice
from .template_heuristics.triton import BaseConfigHeuristic

class Sortable(typing.Protocol):
    def __lt__(self, other: typing.Self) -> bool: ...

class InductorChoices:
    def get_config_heuristics(
        self, device_type: str | None = ...
    ) -> BaseConfigHeuristic: ...
    def get_conv_configs(
        self, device_type: str | None = ...
    ) -> partial[Generator[TritonConfig]]: ...
    def get_flex_attention_fwd_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: str | None = ...
    ) -> list[Any]: ...
    def get_flex_attention_bwd_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: str | None = ...
    ) -> list[Any]: ...
    def get_flex_decode_configs(
        self, head_dim: int, dtype: torch.dtype, device_type: str | None = ...
    ) -> list[Any]: ...
    def get_mm_configs(
        self,
        kernel_inputs: KernelInputs,
        layout: Any,
        templates: list[KernelTemplate | ExternKernelChoice],
        op_name: str,
        kwarg_overrides: dict[str, dict[str, Any]] | None = ...,
    ) -> Generator[ChoiceCaller]: ...
    def triton_kernel_kwargs(
        self,
        kernel_cls: type[TritonKernel],
        features: SIMDKernelFeatures,
        groups: list[sympy.Expr],
        kernel_kwargs: dict[str, Any],
    ) -> dict[str, Any]: ...
    @staticmethod
    def should_use_cooperative_reduction(features: SIMDKernelFeatures) -> bool: ...
    @staticmethod
    def should_use_persistent_reduction(
        features: SIMDKernelFeatures, cooperative_reduction: bool
    ) -> bool: ...
    @staticmethod
    def reduction_split_factor(
        device: torch.device,
        reduction_numel_hint: int,
        numel_hint: int,
        inner_reduction: bool,
    ) -> int: ...
    @staticmethod
    def can_fuse(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool: ...
    @staticmethod
    def can_fuse_vertical(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool: ...
    @staticmethod
    def can_fuse_horizontal(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool: ...
    @staticmethod
    def score_fusion(
        scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> Sortable: ...
