from ...configuration_utils import PretrainedConfig

"""XLNet configuration"""
logger = ...

class XLNetConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        d_model=...,
        n_layer=...,
        n_head=...,
        d_inner=...,
        ff_activation=...,
        untie_r=...,
        attn_type=...,
        initializer_range=...,
        layer_norm_eps=...,
        dropout=...,
        mem_len=...,
        reuse_len=...,
        use_mems_eval=...,
        use_mems_train=...,
        bi_data=...,
        clamp_len=...,
        same_length=...,
        summary_type=...,
        summary_use_proj=...,
        summary_activation=...,
        summary_last_dropout=...,
        start_n_top=...,
        end_n_top=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...
    @property
    def max_position_embeddings(self):  # -> Literal[-1]:
        ...
    @max_position_embeddings.setter
    def max_position_embeddings(self, value): ...

__all__ = ["XLNetConfig"]
