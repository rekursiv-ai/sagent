from collections.abc import Mapping
from typing import Any

from ... import PreTrainedTokenizer, TensorType
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec

"""OpenAI GPT-2 configuration"""
logger = ...

class GPT2Config(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        n_positions=...,
        n_embd=...,
        n_layer=...,
        n_head=...,
        n_inner=...,
        activation_function=...,
        resid_pdrop=...,
        embd_pdrop=...,
        attn_pdrop=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        summary_type=...,
        summary_use_proj=...,
        summary_activation=...,
        summary_proj_to_labels=...,
        summary_first_dropout=...,
        scale_attn_weights=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        scale_attn_by_inverse_layer_idx=...,
        reorder_and_upcast_attn=...,
        **kwargs,
    ) -> None: ...

class GPT2OnnxConfig(OnnxConfigWithPast):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = ...,
        patching_specs: list[PatchingSpec] | None = ...,
        use_past: bool = ...,
    ) -> None: ...
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def num_attention_heads(self) -> int: ...
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = ...,
        seq_length: int = ...,
        is_pair: bool = ...,
        framework: TensorType | None = ...,
    ) -> Mapping[str, Any]: ...
    @property
    def default_onnx_opset(self) -> int: ...

__all__ = ["GPT2Config", "GPT2OnnxConfig"]
