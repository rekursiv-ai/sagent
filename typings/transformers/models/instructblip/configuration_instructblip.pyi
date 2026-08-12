from ...configuration_utils import PretrainedConfig

"""InstructBLIP model configuration"""
logger = ...

class InstructBlipVisionConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=...,
        intermediate_size=...,
        num_hidden_layers=...,
        num_attention_heads=...,
        image_size=...,
        patch_size=...,
        hidden_act=...,
        layer_norm_eps=...,
        attention_dropout=...,
        initializer_range=...,
        qkv_bias=...,
        **kwargs,
    ) -> None: ...

class InstructBlipQFormerConfig(PretrainedConfig):
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
        position_embedding_type=...,
        cross_attention_frequency=...,
        encoder_hidden_size=...,
        **kwargs,
    ) -> None: ...

class InstructBlipConfig(PretrainedConfig):
    def __init__(
        self,
        vision_config=...,
        qformer_config=...,
        text_config=...,
        num_query_tokens=...,
        image_token_index=...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def from_vision_qformer_text_configs(
        cls,
        vision_config: InstructBlipVisionConfig,
        qformer_config: InstructBlipQFormerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):  # -> Self:
        ...

__all__ = [
    "InstructBlipConfig",
    "InstructBlipQFormerConfig",
    "InstructBlipVisionConfig",
]
