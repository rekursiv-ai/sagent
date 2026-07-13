from collections.abc import Mapping

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

"""XLM configuration"""
logger = ...

class XLMConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        emb_dim=...,
        n_layers=...,
        n_heads=...,
        dropout=...,
        attention_dropout=...,
        gelu_activation=...,
        sinusoidal_embeddings=...,
        causal=...,
        asm=...,
        n_langs=...,
        use_lang_emb=...,
        max_position_embeddings=...,
        embed_init_std=...,
        layer_norm_eps=...,
        init_std=...,
        bos_index=...,
        eos_index=...,
        pad_index=...,
        unk_index=...,
        mask_index=...,
        is_encoder=...,
        summary_type=...,
        summary_use_proj=...,
        summary_activation=...,
        summary_proj_to_labels=...,
        summary_first_dropout=...,
        start_n_top=...,
        end_n_top=...,
        mask_token_id=...,
        lang_id=...,
        pad_token_id=...,
        bos_token_id=...,
        **kwargs,
    ) -> None: ...

class XLMOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]: ...

__all__ = ["XLMConfig", "XLMOnnxConfig"]
