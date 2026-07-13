from ...configuration_utils import PretrainedConfig

class Lfm2Config(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    def __init__(
        self,
        vocab_size: int = ...,
        hidden_size: int = ...,
        intermediate_size: int = ...,
        num_hidden_layers: int = ...,
        num_attention_heads: int = ...,
        num_key_value_heads: int = ...,
        max_position_embeddings: int = ...,
        initializer_range: float = ...,
        norm_eps: float = ...,
        use_cache: bool = ...,
        pad_token_id: int = ...,
        bos_token_id: int = ...,
        eos_token_id: int = ...,
        tie_word_embeddings: bool = ...,
        rope_theta: float = ...,
        conv_bias: bool = ...,
        conv_L_cache: int = ...,
        block_multiple_of: int = ...,
        block_ffn_dim_multiplier: float = ...,
        block_auto_adjust_ff_dim: bool = ...,
        full_attn_idxs: list[int] | None = ...,
        layer_types: list[str] | None = ...,
        **kwargs,
    ) -> None: ...

__all__ = ["Lfm2Config"]
