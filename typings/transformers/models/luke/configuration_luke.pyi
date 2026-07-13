from ...configuration_utils import PretrainedConfig

"""LUKE configuration"""
logger = ...

class LukeConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        vocab_size=...,
        entity_vocab_size=...,
        hidden_size=...,
        entity_emb_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        intermediate_size=...,
        hidden_act=...,
        hidden_dropout_prob=...,
        attention_probs_dropout_prob=...,
        max_position_embeddings=...,
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        use_entity_aware_attention=...,
        classifier_dropout=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["LukeConfig"]
