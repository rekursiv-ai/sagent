from ...configuration_utils import PretrainedConfig

"""XGLM model configuration"""
logger = ...

class XGLMConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        max_position_embeddings=...,
        d_model=...,
        ffn_dim=...,
        num_layers=...,
        attention_heads=...,
        activation_function=...,
        dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        layerdrop=...,
        init_std=...,
        scale_embedding=...,
        use_cache=...,
        decoder_start_token_id=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["XGLMConfig"]
