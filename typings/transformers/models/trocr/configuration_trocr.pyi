from ...configuration_utils import PretrainedConfig

"""TrOCR model configuration"""
logger = ...

class TrOCRConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        d_model=...,
        decoder_layers=...,
        decoder_attention_heads=...,
        decoder_ffn_dim=...,
        activation_function=...,
        max_position_embeddings=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        decoder_start_token_id=...,
        init_std=...,
        decoder_layerdrop=...,
        use_cache=...,
        scale_embedding=...,
        use_learned_position_embeddings=...,
        layernorm_embedding=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["TrOCRConfig"]
