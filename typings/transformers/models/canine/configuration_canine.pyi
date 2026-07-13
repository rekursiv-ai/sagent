from ...configuration_utils import PretrainedConfig

"""CANINE model configuration"""
logger = ...

class CanineConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
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
        downsampling_rate=...,
        upsampling_kernel_size=...,
        num_hash_functions=...,
        num_hash_buckets=...,
        local_transformer_stride=...,
        **kwargs,
    ) -> None: ...

__all__ = ["CanineConfig"]
