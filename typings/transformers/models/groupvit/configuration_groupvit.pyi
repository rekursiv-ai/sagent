from collections.abc import Mapping
from typing import Any

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...processing_utils import ProcessorMixin
from ...utils import TensorType

"""GroupViT model configuration"""
logger = ...

class GroupViTTextConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        max_position_embeddings=...,
        hidden_act=...,
        layer_norm_eps=...,
        dropout=...,
        attention_dropout=...,
        initializer_range=...,
        initializer_factor=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

class GroupViTVisionConfig(PretrainedConfig):
    model_type = ...
    base_config_key = ...
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        depths=...,
        num_hidden_layers=...,
        num_group_tokens=...,
        num_output_groups=...,
        num_attention_heads=...,
        image_size=...,
        patch_size=...,
        num_channels=...,
        hidden_act=...,
        layer_norm_eps=...,
        dropout=...,
        attention_dropout=...,
        initializer_range=...,
        initializer_factor=...,
        assign_eps=...,
        assign_mlp_ratio=...,
        **kwargs,
    ) -> None: ...

class GroupViTConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    def __init__(
        self,
        text_config=...,
        vision_config=...,
        projection_dim=...,
        projection_intermediate_dim=...,
        logit_scale_init_value=...,
        **kwargs,
    ) -> None: ...

class GroupViTOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...
    def generate_dummy_inputs(
        self,
        processor: ProcessorMixin,
        batch_size: int = ...,
        seq_length: int = ...,
        framework: TensorType | None = ...,
    ) -> Mapping[str, Any]: ...
    @property
    def default_onnx_opset(self) -> int: ...

__all__ = [
    "GroupViTConfig",
    "GroupViTOnnxConfig",
    "GroupViTTextConfig",
    "GroupViTVisionConfig",
]
