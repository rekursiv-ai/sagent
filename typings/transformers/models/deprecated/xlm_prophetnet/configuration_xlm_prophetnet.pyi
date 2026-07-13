from collections.abc import Callable

from ....configuration_utils import PretrainedConfig

"""XLM-ProphetNet model configuration"""
logger = ...

class XLMProphetNetConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        activation_dropout: float | None = ...,
        activation_function: str | Callable | None = ...,
        vocab_size: int | None = ...,
        hidden_size: int | None = ...,
        encoder_ffn_dim: int | None = ...,
        num_encoder_layers: int | None = ...,
        num_encoder_attention_heads: int | None = ...,
        decoder_ffn_dim: int | None = ...,
        num_decoder_layers: int | None = ...,
        num_decoder_attention_heads: int | None = ...,
        attention_dropout: float | None = ...,
        dropout: float | None = ...,
        max_position_embeddings: int | None = ...,
        init_std: float | None = ...,
        is_encoder_decoder: bool | None = ...,
        add_cross_attention: bool | None = ...,
        decoder_start_token_id: int | None = ...,
        ngram: int | None = ...,
        num_buckets: int | None = ...,
        relative_max_distance: int | None = ...,
        disable_ngram_loss: bool | None = ...,
        eps: float | None = ...,
        use_cache: bool | None = ...,
        pad_token_id: int | None = ...,
        bos_token_id: int | None = ...,
        eos_token_id: int | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def num_hidden_layers(self) -> int: ...
    @num_hidden_layers.setter
    def num_hidden_layers(self, value): ...

__all__ = ["XLMProphetNetConfig"]
