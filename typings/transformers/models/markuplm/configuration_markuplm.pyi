from ...configuration_utils import PretrainedConfig

"""MarkupLM model configuration"""
logger = ...

class MarkupLMConfig(PretrainedConfig):
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
        type_vocab_size=...,
        initializer_range=...,
        layer_norm_eps=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        max_xpath_tag_unit_embeddings=...,
        max_xpath_subs_unit_embeddings=...,
        tag_pad_id=...,
        subs_pad_id=...,
        xpath_unit_hidden_size=...,
        max_depth=...,
        use_cache=...,
        classifier_dropout=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MarkupLMConfig"]
