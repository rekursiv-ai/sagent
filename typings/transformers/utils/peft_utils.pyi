import os

ADAPTER_CONFIG_NAME = ...
ADAPTER_WEIGHTS_NAME = ...
ADAPTER_SAFE_WEIGHTS_NAME = ...

def find_adapter_config_file(
    model_id: str,
    cache_dir: str | os.PathLike | None = ...,
    force_download: bool = ...,
    resume_download: bool | None = ...,
    proxies: dict[str, str] | None = ...,
    token: bool | str | None = ...,
    revision: str | None = ...,
    local_files_only: bool = ...,
    subfolder: str = ...,
    _commit_hash: str | None = ...,
) -> str | None: ...
def check_peft_version(min_version: str) -> None: ...
