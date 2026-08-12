from collections.abc import Mapping
from typing import Any

from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import FeatureExtractionMixin
from ...onnx import OnnxSeq2SeqConfigWithPast
from ...tokenization_utils_base import PreTrainedTokenizerBase
from ...utils import TensorType

"""Whisper model configuration"""
logger = ...
NON_SPEECH_TOKENS = ...
NON_SPEECH_TOKENS_MULTI = ...

class WhisperConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size=...,
        num_mel_bins=...,
        encoder_layers=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_attention_heads=...,
        decoder_ffn_dim=...,
        encoder_ffn_dim=...,
        encoder_layerdrop=...,
        decoder_layerdrop=...,
        decoder_start_token_id=...,
        use_cache=...,
        is_encoder_decoder=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        scale_embedding=...,
        max_source_positions=...,
        max_target_positions=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        suppress_tokens=...,
        begin_suppress_tokens=...,
        use_weighted_layer_sum=...,
        classifier_proj_size=...,
        apply_spec_augment=...,
        mask_time_prob=...,
        mask_time_length=...,
        mask_time_min_masks=...,
        mask_feature_prob=...,
        mask_feature_length=...,
        mask_feature_min_masks=...,
        median_filter_width=...,
        **kwargs,
    ) -> None: ...

class WhisperOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    def generate_dummy_inputs(
        self,
        preprocessor: PreTrainedTokenizerBase | FeatureExtractionMixin,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
        sampling_rate: int = ...,
        time_duration: float = ...,
        frequency: int = ...,
    ) -> Mapping[str, Any]: ...
    @property
    def atol_for_validation(self) -> float: ...

__all__ = ["WhisperConfig", "WhisperOnnxConfig"]
