from ...configuration_utils import PretrainedConfig

"""MAMBA2 configuration"""
logger = ...

class Mamba2Config(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
        num_heads=...,
        head_dim=...,
        vocab_size=...,
        hidden_size=...,
        state_size=...,
        num_hidden_layers=...,
        layer_norm_epsilon=...,
        pad_token_id=...,
        bos_token_id=...,
        eos_token_id=...,
        expand=...,
        conv_kernel=...,
        n_groups=...,
        use_bias=...,
        use_conv_bias=...,
        hidden_act=...,
        initializer_range=...,
        residual_in_fp32=...,
        time_step_rank=...,
        time_step_min=...,
        time_step_max=...,
        time_step_floor=...,
        time_step_limit=...,
        rescale_prenorm_residual=...,
        use_cache=...,
        rms_norm=...,
        chunk_size=...,
        tie_word_embeddings=...,
        **kwargs,
    ) -> None: ...

__all__ = ["Mamba2Config"]
