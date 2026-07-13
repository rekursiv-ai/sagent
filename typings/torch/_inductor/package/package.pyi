from torch.export.pt2_archive._package import AOTI_FILES, AOTICompiledModel
from torch.types import FileLike

log = ...

def compile_so(aoti_dir: str, aoti_files: list[str], so_path: str) -> str: ...
def package_aoti(archive_file: FileLike, aoti_files: AOTI_FILES) -> FileLike: ...
def load_package(
    path: FileLike,
    model_name: str = ...,
    run_single_threaded: bool = ...,
    num_runners: int = ...,
    device_index: int = ...,
) -> AOTICompiledModel: ...
