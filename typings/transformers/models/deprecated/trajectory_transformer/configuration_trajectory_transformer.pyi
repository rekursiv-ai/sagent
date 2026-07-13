from ....configuration_utils import PretrainedConfig

"""TrajectoryTransformer model configuration"""
logger = ...

class TrajectoryTransformerConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        action_weight=...,
        reward_weight=...,
        value_weight=...,
        block_size=...,
        action_dim=...,
        observation_dim=...,
        transition_dim=...,
        n_layer=...,
        n_head=...,
        n_embd=...,
        embd_pdrop=...,
        attn_pdrop=...,
        resid_pdrop=...,
        learning_rate=...,
        max_position_embeddings=...,
        initializer_range=...,
        layer_norm_eps=...,
        kaiming_initializer_range=...,
        use_cache=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        **kwargs,
    ) -> None: ...

__all__ = ["TrajectoryTransformerConfig"]
