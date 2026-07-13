from collections.abc import Iterable

import enum

from ..configuration_utils import PretrainedConfig

"""Collection of utils to be used by backbones and their components."""

class BackboneType(enum.Enum):
    TIMM = ...
    TRANSFORMERS = ...

def verify_out_features_out_indices(
    out_features: Iterable[str] | None,
    out_indices: Iterable[int] | None,
    stage_names: Iterable[str] | None,
):  # -> None:
    ...
def get_aligned_output_features_output_indices(
    out_features: list[str] | None,
    out_indices: list[int] | tuple[int] | None,
    stage_names: list[str],
) -> tuple[list[str], list[int]]: ...

class BackboneMixin:
    backbone_type: BackboneType | None = ...
    has_attentions: bool = ...
    @property
    def out_features(self):  # -> list[str]:
        ...
    @out_features.setter
    def out_features(self, out_features: list[str]):  # -> None:
        ...
    @property
    def out_indices(self):  # -> list[Any] | list[int]:
        ...
    @out_indices.setter
    def out_indices(self, out_indices: tuple[int] | list[int]):  # -> None:
        ...
    @property
    def out_feature_channels(self):  # -> dict[Any, Any]:
        ...
    @property
    def channels(self):  # -> list[Any]:
        ...
    def forward_with_filtered_kwargs(self, *args, **kwargs): ...
    def forward(
        self,
        pixel_values,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
    ): ...
    def to_dict(self): ...

class BackboneConfigMixin:
    @property
    def out_features(self):  # -> list[str]:
        ...
    @out_features.setter
    def out_features(self, out_features: list[str]):  # -> None:
        ...
    @property
    def out_indices(self):  # -> list[int]:
        ...
    @out_indices.setter
    def out_indices(self, out_indices: tuple[int, ...] | list[int]):  # -> None:
        ...
    def to_dict(self): ...

def load_backbone(config):  # -> Any:
    ...
def verify_backbone_config_arguments(
    use_timm_backbone: bool,
    use_pretrained_backbone: bool,
    backbone: str | None,
    backbone_config: dict | PretrainedConfig | None,
    backbone_kwargs: dict | None,
):  # -> None:
    ...
