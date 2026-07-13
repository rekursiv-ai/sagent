from ...configuration_utils import PretrainedConfig

"""RWKV configuration"""
logger = ...

class RwkvConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        context_length=...,
        hidden_size=...,
        num_hidden_layers=...,
        attention_hidden_size=...,
        intermediate_size=...,
        layer_norm_epsilon=...,
        bos_token_id=...,
        eos_token_id=...,
        rescale_every=...,
        tie_word_embeddings=...,
        use_cache=...,
        **kwargs,
    ) -> None: ...

__all__ = ["RwkvConfig"]
