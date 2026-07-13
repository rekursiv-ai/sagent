from ...configuration_utils import PretrainedConfig

"""BioGPT model configuration"""
logger = ...

class BioGptConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        hidden_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        scale_embedding=...,
        use_cache=...,
        layerdrop=...,
        activation_dropout=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["BioGptConfig"]
