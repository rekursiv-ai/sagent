from collections.abc import Mapping
from typing import Any

from ... import PreTrainedTokenizerBase, TensorType
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

logger = ...

class VisionEncoderDecoderConfig(PretrainedConfig):
    model_type = ...
    sub_configs = ...
    has_no_defaults_at_init = ...
    def __init__(self, **kwargs) -> None: ...
    @classmethod
    def from_encoder_decoder_configs(
        cls,
        encoder_config: PretrainedConfig,
        decoder_config: PretrainedConfig,
        **kwargs,
    ) -> PretrainedConfig: ...

class VisionEncoderDecoderEncoderOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def atol_for_validation(self) -> float: ...
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]: ...

class VisionEncoderDecoderDecoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
    ) -> Mapping[str, Any]: ...

class VisionEncoderDecoderOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> None: ...
    def get_encoder_config(self, encoder_config: PretrainedConfig) -> OnnxConfig: ...
    def get_decoder_config(
        self,
        encoder_config: PretrainedConfig,
        decoder_config: PretrainedConfig,
        feature: str = ...,
    ) -> OnnxConfig: ...

__all__ = ["VisionEncoderDecoderConfig", "VisionEncoderDecoderOnnxConfig"]
