from collections import OrderedDict

import os

from .configuration_auto import replace_list_option_in_docstrings
from ...utils.import_utils import requires

"""AutoVideoProcessor class."""
logger = ...
VIDEO_PROCESSOR_MAPPING_NAMES: OrderedDict[str, tuple[str | None, str | None]] = ...
VIDEO_PROCESSOR_MAPPING = ...

def video_processor_class_from_name(class_name: str):  # -> Any | None:
    ...
def get_video_processor_config(
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

@requires(backends=("vision", "torchvision"))
class AutoVideoProcessor:
    def __init__(self) -> None: ...
    @classmethod
    @replace_list_option_in_docstrings(VIDEO_PROCESSOR_MAPPING_NAMES)
    def from_pretrained(
        cls, pretrained_model_name_or_path, *inputs, **kwargs
    ):  # -> Any:
        ...
    @staticmethod
    def register(config_class, video_processor_class, exist_ok=...):  # -> None:
        ...

__all__ = ["VIDEO_PROCESSOR_MAPPING", "AutoVideoProcessor"]
