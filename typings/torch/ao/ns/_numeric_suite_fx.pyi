from collections.abc import Callable
from typing import Any

from torch import Tensor, nn
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from torch.ao.quantization import quantize_fx
from torch.ao.quantization.backend_config import BackendConfig
from torch.fx import GraphModule

import torch

from .fx.ns_types import NSNodeTargetType, NSResultsType

"""
This module contains tooling to compare weights and activations
across models. Example usage::

    import copy
    import torch
    import torch.ao.quantization.quantize_fx as quantize_fx
    import torch.ao.ns._numeric_suite_fx as ns

    m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1)).eval()
    mp = quantize_fx.prepare_fx(m, {"": torch.ao.quantization.default_qconfig})
    # We convert a copy because we need the original prepared model
    # to be available for comparisons, and `quantize_fx.convert_fx` is inplace.
    mq = quantize_fx.convert_fx(copy.deepcopy(mp))

    #
    # Comparing weights
    #

    # extract weight pairs
    weight_comparison = ns.extract_weights("a", mp, "b", mq)

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        weight_comparison, "a", "b", torch.ao.ns.fx.utils.compute_sqnr, "sqnr"
    )

    # weight_comparison contains the weights from `mp` and `mq` stored
    # in pairs, and can be used for further analysis.


    #
    # Comparing activations, with error propagation
    #

    # add loggers
    mp_ns, mq_ns = ns.add_loggers(
        "a", copy.deepcopy(mp), "b", copy.deepcopy(mq), ns.OutputLogger
    )

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_ns(datum)
    mq_ns(datum)

    # extract intermediate activations
    act_comparison = ns.extract_logger_info(mp_ns, mq_ns, ns.OutputLogger, "b")

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        act_comparison, "a", "b", torch.ao.ns.fx.utils.compute_sqnr, "sqnr"
    )

    # act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

    #
    # Comparing activations, without error propagation
    #

    # create shadow model
    mp_shadows_mq = ns.add_shadow_loggers(
        "a", copy.deepcopy(mp), "b", copy.deepcopy(mq), ns.OutputLogger
    )

    # send an example datum to capture intermediate activations
    datum = torch.randn(1, 1, 1, 1)
    mp_shadows_mq(datum)

    # extract intermediate activations
    shadow_act_comparison = ns.extract_shadow_logger_info(
        mp_shadows_mq, ns.OutputLogger, "b"
    )

    # add SQNR for each comparison, inplace
    ns.extend_logger_results_with_comparison(
        shadow_act_comparison, "a", "b", torch.ao.ns.fx.utils.compute_sqnr, "sqnr"
    )

    # shadow_act_comparison contains the activations from `mp_ns` and `mq_ns` stored
    # in pairs, and can be used for further analysis.

"""
type RNNReturnType = tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

class OutputLogger(nn.Module):
    stats: list[torch.Tensor]
    stats_rnn: list[RNNReturnType]
    _is_impure = ...
    def __init__(
        self,
        ref_node_name: str,
        prev_node_name: str,
        model_name: str,
        ref_name: str,
        prev_node_target_type: str,
        ref_node_target_type: str,
        results_type: str,
        index_within_arg: int,
        index_of_arg: int,
        fqn: str | None,
        qconfig_str: str | None = ...,
    ) -> None: ...
    def forward(self, x) -> Tensor | tuple[Any, Any] | tuple[Any, ...]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Tensor | tuple[Any, Any] | tuple[Any, ...]: ...

class OutputComparisonLogger(OutputLogger):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, x, x_ref) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

class NSTracer(quantize_fx.QuantizationTracer):
    def is_leaf_module(
        self, m: torch.nn.Module, module_qualified_name: str
    ) -> bool: ...

def extract_weights(
    model_name_a: str,
    model_a: nn.Module,
    model_name_b: str,
    model_b: nn.Module,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = ...,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = ...,
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]]
    | None = ...,
) -> NSResultsType: ...
def add_loggers(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs: bool = ...,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = ...,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = ...,
) -> tuple[nn.Module, nn.Module]: ...
def extract_logger_info(
    model_a: nn.Module,
    model_b: nn.Module,
    logger_cls: Callable,
    model_name_to_use_for_layer_names: str,
) -> NSResultsType: ...
def add_shadow_loggers(
    name_a: str,
    model_a: nn.Module,
    name_b: str,
    model_b: nn.Module,
    logger_cls: Callable,
    should_log_inputs: bool = ...,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = ...,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = ...,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = ...,
) -> nn.Module: ...
def extract_shadow_logger_info(
    model_a_shadows_b: nn.Module,
    logger_cls: Callable,
    model_name_to_use_for_layer_names: str,
) -> NSResultsType: ...
def extend_logger_results_with_comparison(
    results: NSResultsType,
    model_name_1: str,
    model_name_2: str,
    comparison_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    comparison_name: str,
) -> None: ...
def prepare_n_shadows_model(
    model: torch.nn.Module,
    example_inputs: Any,
    qconfig_multi_mapping: QConfigMultiMapping,
    backend_config: BackendConfig,
    custom_prepare_fn: Callable | None = ...,
    custom_prepare_kwargs: dict[str, Any] | None = ...,
    custom_tracer: Any = ...,
) -> GraphModule: ...
def loggers_set_enabled(model: torch.nn.Module, enabled: bool) -> None: ...
def loggers_set_save_activations(
    model: torch.nn.Module, save_activations: bool
) -> None: ...
def convert_n_shadows_model(
    model: GraphModule,
    custom_convert_fn: Callable | None = ...,
    custom_convert_kwargs: dict[str, Any] | None = ...,
) -> GraphModule: ...
def extract_results_n_shadows_model(model: torch.nn.Module) -> NSResultsType: ...
def print_comparisons_n_shadows_model(results: NSResultsType) -> None: ...
