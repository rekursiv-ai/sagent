from ...configuration_utils import PretrainedConfig

"""Configuration for Backbone models"""
logger = ...

class TimmBackboneConfig(PretrainedConfig):
    def __init__(
        self,
        backbone=...,
        num_channels=...,
        features_only=...,
        use_pretrained_backbone=...,
        out_indices=...,
        freeze_batch_norm_2d=...,
        **kwargs,
    ) -> None: ...

__all__ = ["TimmBackboneConfig"]
