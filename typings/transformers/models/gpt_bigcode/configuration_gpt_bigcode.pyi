from ...configuration_utils import PretrainedConfig

"""GPTBigCode configuration"""
logger = ...

class GPTBigCodeConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        n_positions=...,
        n_embd=...,
        n_layer=...,
        n_head=...,
        n_inner=...,
        activation_function=...,
        resid_pdrop=...,
        embd_pdrop=...,
        attn_pdrop=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        scale_attn_weights=...,
        use_cache=...,
        bos_token_id=...,
        eos_token_id=...,
        attention_softmax_in_fp32=...,
        scale_attention_softmax_in_fp32=...,
        multi_query=...,
        **kwargs,
    ) -> None: ...

__all__ = ["GPTBigCodeConfig"]
