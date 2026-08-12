from typing import Any
from torch import nn
from torch._inductor import config as inductor_config

import torch

from .pre_grad import efficient_conv_bn_eval_pass
from ..pattern_matcher import (
    CallFunctionVarArgs,
    CallModuleVarArgs,
    Match,
    register_graph_pattern,
)

def efficient_conv_bn_eval(
    bn: nn.modules.batchnorm._BatchNorm, conv: nn.modules.conv._ConvNd, x: torch.Tensor
) -> Any: ...
def efficient_conv_bn_eval_decomposed(
    bn_weight,
    bn_bias,
    bn_running_mean,
    bn_running_var,
    bn_eps,
    conv: torch._ops.OpOverload,
    conv_weight,
    conv_bias,
    x,
    conv_remainging_args,
) -> Any: ...
@register_graph_pattern(
    CallFunctionVarArgs([torch.nn.functional.batch_norm]),
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: (
        not inductor_config.freezing
        and inductor_config.efficient_conv_bn_eval_fx_passes
    ),
)
def efficient_conv_bn_eval_graph_transform_inlined(
    match: Match, *args, **kwargs
) -> None: ...
@register_graph_pattern(
    CallFunctionVarArgs([torch.ops.aten.batch_norm.default]),
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: (
        not inductor_config.freezing
        and inductor_config.efficient_conv_bn_eval_fx_passes
    ),
)
def efficient_conv_bn_eval_graph_transform_decomposed(
    match: Match, *args, **kwargs
) -> None: ...
@register_graph_pattern(
    CallModuleVarArgs(
        [
            nn.modules.batchnorm._BatchNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
        ]
    ),
    pass_dict=efficient_conv_bn_eval_pass,
    extra_check=lambda match: (
        not inductor_config.freezing
        and inductor_config.efficient_conv_bn_eval_fx_passes
    ),
)
def efficient_conv_bn_eval_graph_transform(match: Match, *args, **kwargs) -> None: ...
