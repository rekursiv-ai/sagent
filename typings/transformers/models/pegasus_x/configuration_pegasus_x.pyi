from ...configuration_utils import PretrainedConfig

"""PEGASUS-X model configuration"""
logger = ...

class PegasusXConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        encoder_layers=...,
        encoder_ffn_dim=...,
        encoder_attention_heads=...,
        decoder_layers=...,
        decoder_ffn_dim=...,
        decoder_attention_heads=...,
        encoder_layerdrop=...,
        decoder_layerdrop=...,
        use_cache=...,
        is_encoder_decoder=...,
        activation_function=...,
        d_model=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        init_std=...,
        decoder_start_token_id=...,
        scale_embedding=...,
        pad_token_id=...,
        eos_token_id=...,
        forced_eos_token_id=...,
        num_global_tokens=...,
        block_size=...,
        stagger_local_blocks=...,
        **kwargs,
    ) -> None: ...
    @property
    def num_attention_heads(self) -> int: ...
    @property
    def hidden_size(self) -> int: ...

__all__ = ["PegasusXConfig"]
