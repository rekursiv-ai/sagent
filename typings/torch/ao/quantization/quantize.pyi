from typing import Any

import typing_extensions

from .utils import DEPRECATION_WARNING

__all__ = [
    "add_quant_dequant",
    "convert",
    "get_default_custom_config_dict",
    "prepare",
    "prepare_qat",
    "propagate_qconfig_",
    "quantize",
    "quantize_dynamic",
    "quantize_qat",
    "swap_module",
]
is_activation_post_process = ...
_DEFAULT_CUSTOM_CONFIG_DICT = ...

def get_default_custom_config_dict() -> dict[
    str, dict[type[LSTM | MultiheadAttention], Any] | dict[Any, Any]
]: ...
def propagate_qconfig_(
    module, qconfig_dict=..., prepare_custom_config_dict=...
) -> None: ...
def add_quant_dequant(module) -> QuantWrapper: ...
@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare(
    model,
    inplace=...,
    allow_list=...,
    observer_non_leaf_module_list=...,
    prepare_custom_config_dict=...,
): ...
@typing_extensions.deprecated(DEPRECATION_WARNING)
def quantize(model, run_fn, run_args, mapping=..., inplace=...): ...
@typing_extensions.deprecated(DEPRECATION_WARNING)
def quantize_dynamic(model, qconfig_spec=..., dtype=..., mapping=..., inplace=...): ...
@typing_extensions.deprecated(DEPRECATION_WARNING)
def prepare_qat(model, mapping=..., inplace=...): ...
@typing_extensions.deprecated(DEPRECATION_WARNING)
def quantize_qat(model, run_fn, run_args, inplace=...): ...
@typing_extensions.deprecated(DEPRECATION_WARNING)
def convert(
    module,
    mapping=...,
    inplace=...,
    remove_qconfig=...,
    is_reference=...,
    convert_custom_config_dict=...,
    use_precomputed_fake_quant=...,
): ...
def swap_module(
    mod, mapping, custom_module_class_mapping, use_precomputed_fake_quant=...
): ...
