from typing import Any

from ...configuration_utils import PretrainedConfig

class Emu3VQVAEConfig(PretrainedConfig):
    def __init__(
        self,
        codebook_size: int = ...,
        embed_dim: int = ...,
        latent_channels: int = ...,
        double_latent: bool = ...,
        in_channels: int = ...,
        out_channels: int = ...,
        temporal_downsample_factor: int = ...,
        base_channels: int = ...,
        channel_multiplier: list[int] = ...,
        num_res_blocks: int = ...,
        attn_resolutions: list[int] = ...,
        hidden_size: int = ...,
        num_attention_heads: int = ...,
        attention_dropout: float = ...,
        **kwargs,
    ) -> None: ...

class Emu3TextConfig(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size: int = ...,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        num_key_value_heads: int | None = ...,
        hidden_act: str = ...,
        max_position_embeddings: int = ...,
        rms_norm_eps: float = ...,
        use_cache: bool = ...,
        pad_token_id: int = ...,
        bos_token_id: int = ...,
        eos_token_id: int = ...,
        tie_word_embeddings: bool = ...,
        rope_theta: float = ...,
        rope_scaling: dict[str, Any] | None = ...,
        mlp_bias=...,
        attention_bias=...,
        attention_dropout: float = ...,
        initializer_range: float = ...,
        **kwargs,
    ) -> None: ...

class Emu3Config(PretrainedConfig):
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vq_config: dict | Emu3VQVAEConfig = ...,
        text_config: dict | Emu3TextConfig = ...,
        vocabulary_map: dict[int, int] | None = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["Emu3Config", "Emu3TextConfig", "Emu3VQVAEConfig"]
