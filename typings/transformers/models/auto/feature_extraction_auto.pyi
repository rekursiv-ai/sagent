import os

from .configuration_auto import replace_list_option_in_docstrings

"""AutoFeatureExtractor class."""
logger = ...
FEATURE_EXTRACTOR_MAPPING_NAMES = ...
FEATURE_EXTRACTOR_MAPPING = ...

def feature_extractor_class_from_name(class_name: str):  # -> Any | None:
    ...
def get_feature_extractor_config(
    pretrained_model_name_or_path: str | os.PathLike,
    cache_dir: str | os.PathLike | None = ...,
    force_download: bool = ...,
    resume_download: bool | None = ...,
    proxies: dict[str, str] | None = ...,
    token: bool | str | None = ...,
    revision: str | None = ...,
    local_files_only: bool = ...,
    **kwargs,
):  # -> dict[Any, Any] | Any:
    ...

class AutoFeatureExtractor:
    def __init__(self) -> None: ...
    @classmethod
    @replace_list_option_in_docstrings(FEATURE_EXTRACTOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):  # -> Any:
        ...
    @staticmethod
    def register(config_class, feature_extractor_class, exist_ok=...):  # -> None:
        ...

__all__ = ["FEATURE_EXTRACTOR_MAPPING", "AutoFeatureExtractor"]
