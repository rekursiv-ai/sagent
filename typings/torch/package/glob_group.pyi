from collections.abc import Iterable

type GlobPattern = str | Iterable[str]

class GlobGroup:
    def __init__(
        self,
        include: GlobPattern,
        *,
        exclude: GlobPattern = ...,
        separator: str = ...,
    ) -> None: ...
    def matches(self, candidate: str) -> bool: ...
