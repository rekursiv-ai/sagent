import dataclasses

log = ...

@dataclasses.dataclass(frozen=True)
class TritonBundleEntry:
    kernel_hash: str
    device: int
    directory: str

@dataclasses.dataclass(frozen=True)
class TritonKernelArtifact:
    filename: str
    payload: bytes = ...

@dataclasses.dataclass(frozen=True)
class StaticallyLaunchedAutotuner:
    cache_key: str
    kernel_name: str
    kernel: CachingAutotuner

@dataclasses.dataclass(frozen=True)
class TritonKernelArtifacts:
    kernel_hash: str
    device: int
    artifacts: list[TritonKernelArtifact]

@dataclasses.dataclass(frozen=True)
class TritonBundlerMetadata:
    cached_kernel_names: list[str]
    statically_launched_kernel_names: list[str]

@dataclasses.dataclass(frozen=True)
class TritonBundle:
    kernel_artifacts: list[TritonKernelArtifacts]
    static_autotuners: list[StaticallyLaunchedAutotuner]

class TritonBundler:
    _entries: list[TritonBundleEntry] | None = ...
    _static_autotuners: list[StaticallyLaunchedAutotuner] | None = ...
    _REPLACE_BYTES: bytes = ...
    @staticmethod
    def is_enabled() -> bool: ...
    @classmethod
    def begin_compile(cls) -> None: ...
    @classmethod
    def end_compile(cls) -> None: ...
    @classmethod
    def put(cls, kernel_hash: str, device: int) -> None: ...
    @classmethod
    def put_static_autotuner(cls, key: str, kernel: CachingAutotuner) -> None: ...
    @classmethod
    def collect_static_autotuners(
        cls,
    ) -> tuple[list[StaticallyLaunchedAutotuner], list[str]]: ...
    @classmethod
    def load_autotuners(
        cls, static_autotuners: list[StaticallyLaunchedAutotuner] | None
    ) -> list[str]: ...
    @classmethod
    def collect(cls) -> tuple[TritonBundle, TritonBundlerMetadata | None]: ...
    @staticmethod
    def read_and_emit(bundle: TritonBundle) -> TritonBundlerMetadata | None: ...
