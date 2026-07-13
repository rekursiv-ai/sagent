from ...configuration_utils import PretrainedConfig

"""Decision Transformer model configuration"""
logger = ...

class DecisionTransformerConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        state_dim=...,
        act_dim=...,
        hidden_size=...,
        max_ep_len=...,
        action_tanh=...,
        vocab_size=...,
        n_positions=...,
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
        scale_attn_by_inverse_layer_idx=...,
        reorder_and_upcast_attn=...,
        **kwargs,
    ) -> None: ...

__all__ = ["DecisionTransformerConfig"]
