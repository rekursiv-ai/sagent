from ...configuration_utils import PretrainedConfig

"""MAMBA configuration"""
logger = ...

class MambaConfig(PretrainedConfig):
    model_type = ...
    def __init__(
        self,
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
        use_bias=...,
        use_conv_bias=...,
        hidden_act=...,
        initializer_range=...,
        residual_in_fp32=...,
        time_step_rank=...,
        time_step_scale=...,
        time_step_min=...,
        time_step_max=...,
        time_step_init_scheme=...,
        time_step_floor=...,
        rescale_prenorm_residual=...,
        use_cache=...,
        use_mambapy=...,
        **kwargs,
    ) -> None: ...

__all__ = ["MambaConfig"]
