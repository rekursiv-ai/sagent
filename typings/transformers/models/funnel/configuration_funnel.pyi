from ...configuration_utils import PretrainedConfig

"""Funnel Transformer model configuration"""
logger = ...

class FunnelConfig(PretrainedConfig):
    model_type = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        block_sizes=...,
        block_repeats=...,
        num_decoder_layers=...,
        d_model=...,
        n_head=...,
        d_head=...,
        d_inner=...,
        hidden_act=...,
        hidden_dropout=...,
        attention_dropout=...,
        activation_dropout=...,
        initializer_range=...,
        initializer_std=...,
        layer_norm_eps=...,
        pooling_type=...,
        attention_type=...,
        separate_cls=...,
        truncate_seq=...,
        pool_q_only=...,
        **kwargs,
    ) -> None: ...
    @property
    def num_hidden_layers(self):  # -> int:
        ...
    @num_hidden_layers.setter
    def num_hidden_layers(self, value): ...
    @property
    def num_blocks(self):  # -> int:
        ...
    @num_blocks.setter
    def num_blocks(self, value): ...

__all__ = ["FunnelConfig"]
