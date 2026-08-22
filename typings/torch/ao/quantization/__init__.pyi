from collections.abc import Callable as Callable
from typing import (
    Any,
    Optional as Optional,
    TypeAliasType as TypeAliasType,
    Union as Union,
)

from torch import Tensor as Tensor

import torch

from .fake_quantize import *
from .fuse_modules import (
    fuse_modules as fuse_modules,
    fuse_modules_qat as fuse_modules_qat,
)
from .fuser_method_mappings import *
from .observer import *
from .pt2e._numeric_debugger import (
    CUSTOM_KEY as CUSTOM_KEY,
    NUMERIC_DEBUG_HANDLE_KEY as NUMERIC_DEBUG_HANDLE_KEY,
    compare_results as compare_results,
    extract_results_from_loggers as extract_results_from_loggers,
    generate_numeric_debug_handle as generate_numeric_debug_handle,
    prepare_for_propagation_comparison as prepare_for_propagation_comparison,
)
from .pt2e.export_utils import (
    _allow_exported_model_train_eval as allow_exported_model_train_eval,
    _move_exported_model_to_eval as move_exported_model_to_eval,
    _move_exported_model_to_train as move_exported_model_to_train,
)
from .qconfig import *
from .qconfig_mapping import *
from .quant_type import *
from .quantization_mappings import *
from .quantize import *
from .quantize_jit import *
from .stubs import *

type ObserverOrFakeQuantize = ObserverBase | FakeQuantizeBase
__all__ = [
    "CUSTOM_KEY",
    "NUMERIC_DEBUG_HANDLE_KEY",
    "AffineQuantizedObserverBase",
    "DeQuantStub",
    "FakeQuantize",
    "FakeQuantizeBase",
    "FixedQParamsFakeQuantize",
    "FixedQParamsObserver",
    "FusedMovingAvgObsFakeQuantize",
    "Granularity",
    "HistogramObserver",
    "MappingType",
    "MatchAllNode",
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "MovingAveragePerChannelMinMaxObserver",
    "NoopObserver",
    "ObserverBase",
    "ObserverOrFakeQuantize",
    "Pattern",
    "PerAxis",
    "PerBlock",
    "PerChannelMinMaxObserver",
    "PerGroup",
    "PerRow",
    "PerTensor",
    "PerToken",
    "PlaceholderObserver",
    "QConfig",
    "QConfigAny",
    "QConfigDynamic",
    "QConfigMapping",
    "QuantStub",
    "QuantType",
    "QuantWrapper",
    "RecordingObserver",
    "ReuseInputObserver",
    "TorchAODType",
    "UniformQuantizationObserverBase",
    "ZeroPointDomain",
    "add_quant_dequant",
    "allow_exported_model_train_eval",
    "compare_results",
    "convert",
    "convert_dynamic_jit",
    "convert_jit",
    "default_affine_fixed_qparams_fake_quant",
    "default_affine_fixed_qparams_observer",
    "default_debug_observer",
    "default_dynamic_fake_quant",
    "default_dynamic_quant_observer",
    "default_embedding_fake_quant",
    "default_embedding_fake_quant_4bit",
    "default_eval_fn",
    "default_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_fixed_qparams_range_0to1_observer",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_neg1to1_observer",
    "default_float_qparams_observer",
    "default_float_qparams_observer_4bit",
    "default_fused_act_fake_quant",
    "default_fused_per_channel_wt_fake_quant",
    "default_fused_wt_fake_quant",
    "default_histogram_fake_quant",
    "default_histogram_observer",
    "default_observer",
    "default_per_channel_weight_fake_quant",
    "default_per_channel_weight_observer",
    "default_placeholder_observer",
    "default_reuse_input_observer",
    "default_symmetric_fixed_qparams_fake_quant",
    "default_symmetric_fixed_qparams_observer",
    "default_weight_fake_quant",
    "default_weight_observer",
    "disable_fake_quant",
    "disable_observer",
    "enable_fake_quant",
    "enable_observer",
    "extract_results_from_loggers",
    "fuse_conv_bn",
    "fuse_conv_bn_jit",
    "fuse_conv_bn_relu",
    "fuse_convtranspose_bn",
    "fuse_linear_bn",
    "fuse_modules",
    "fuse_modules_qat",
    ...,
    "fused_wt_fake_quant_range_neg_127_to_127",
    "generate_numeric_debug_handle",
    "get_block_size",
    "get_combined_dict",
    "get_default_compare_output_module_list",
    "get_default_custom_config_dict",
    "get_default_dynamic_quant_module_mappings",
    "get_default_dynamic_sparse_quant_module_mappings",
    "get_default_float_to_quantized_operator_mappings",
    "get_default_qat_module_mappings",
    "get_default_qat_qconfig",
    "get_default_qat_qconfig_dict",
    "get_default_qat_qconfig_mapping",
    "get_default_qconfig",
    "get_default_qconfig_dict",
    "get_default_qconfig_mapping",
    "get_default_qconfig_propagation_list",
    "get_default_static_quant_module_mappings",
    "get_default_static_quant_reference_module_mappings",
    "get_default_static_sparse_quant_module_mappings",
    "get_dynamic_quant_module_class",
    "get_embedding_qat_module_mappings",
    "get_embedding_static_quant_module_mappings",
    "get_fuser_method",
    "get_fuser_method_new",
    "get_observer_state_dict",
    "get_quantized_operator",
    "get_static_quant_module_class",
    "load_observer_state_dict",
    "move_exported_model_to_eval",
    "move_exported_model_to_train",
    "no_observer_set",
    "per_channel_weight_observer_range_neg_127_to_127",
    "prepare",
    "prepare_dynamic_jit",
    "prepare_for_propagation_comparison",
    "prepare_jit",
    "prepare_qat",
    "propagate_qconfig_",
    "qconfig_equals",
    "quantize",
    "quantize_dynamic",
    "quantize_dynamic_jit",
    "quantize_jit",
    "quantize_qat",
    "script_qconfig",
    "script_qconfig_dict",
    "swap_module",
    "weight_observer_range_neg_127_to_127",
]

def default_eval_fn(model, calib_data) -> None: ...

class _DerivedObserverOrFakeQuantize(ObserverBase):
    def __init__(
        self,
        dtype: torch.dtype,
        obs_or_fqs: list[ObserverOrFakeQuantize],
        derive_qparams_fn: Callable[
            [list[ObserverOrFakeQuantize]], tuple[Tensor, Tensor]
        ],
        quant_min: int | None = ...,
        quant_max: int | None = ...,
        qscheme: torch.qscheme | None = ...,
        ch_axis: int | None = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...
    def calculate_qparams(self) -> tuple[Tensor, Tensor]: ...
