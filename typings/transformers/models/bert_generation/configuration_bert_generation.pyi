from ...configuration_utils import PretrainedConfig

"""BertGeneration model configuration"""

class BertGenerationConfig(PretrainedConfig):
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
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        position_embedding_type=...,
        use_cache=...,
        **kwargs,
    ) -> None: ...

__all__ = ["BertGenerationConfig"]
