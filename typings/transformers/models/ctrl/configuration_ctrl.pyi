from ...configuration_utils import PretrainedConfig

"""Salesforce CTRL configuration"""
logger = ...

class CTRLConfig(PretrainedConfig):
    model_type = ...
    keys_to_ignore_at_inference = ...
    attribute_map = ...
    def __init__(
        self,
        vocab_size=...,
        n_positions=...,
        n_embd=...,
        dff=...,
        n_layer=...,
        n_head=...,
        resid_pdrop=...,
        embd_pdrop=...,
        layer_norm_epsilon=...,
        initializer_range=...,
        use_cache=...,
        **kwargs,
    ) -> None: ...

__all__ = ["CTRLConfig"]
