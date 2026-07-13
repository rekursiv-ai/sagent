from ....configuration_utils import PretrainedConfig

"""Transformer XL configuration"""
logger = ...

class TransfoXLConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        cutoffs=...,
        d_model=...,
        d_embed=...,
        n_head=...,
        d_head=...,
        d_inner=...,
        div_val=...,
        pre_lnorm=...,
        n_layer=...,
        mem_len=...,
        clamp_len=...,
        same_length=...,
        proj_share_all_but_first=...,
        attn_type=...,
        sample_softmax=...,
        adaptive=...,
        dropout=...,
        dropatt=...,
        untie_r=...,
        init=...,
        init_range=...,
        proj_init_std=...,
        init_std=...,
        layer_norm_epsilon=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...
    @property
    def max_position_embeddings(self):  # -> Literal[-1]:
        ...
    @max_position_embeddings.setter
    def max_position_embeddings(self, value): ...

__all__ = ["TransfoXLConfig"]
