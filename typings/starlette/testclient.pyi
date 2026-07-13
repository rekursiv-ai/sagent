from typing import Any, Self

import httpx

class TestClient(httpx.Client):
    app: Any
    def __init__(
        self,
        app: Any,
        base_url: str = ...,
        *,
        follow_redirects: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def __enter__(self) -> Self: ...
