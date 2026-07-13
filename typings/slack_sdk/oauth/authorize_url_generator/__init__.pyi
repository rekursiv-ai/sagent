from collections.abc import Sequence

class AuthorizeUrlGenerator:
    def __init__(
        self,
        *,
        client_id: str,
        redirect_uri: str | None = ...,
        scopes: Sequence[str] | None = ...,
        user_scopes: Sequence[str] | None = ...,
        authorization_url: str = ...,
    ) -> None: ...
    def generate(self, state: str, team: str | None = ...) -> str: ...

class OpenIDConnectAuthorizeUrlGenerator:
    def __init__(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        scopes: Sequence[str] | None = ...,
        authorization_url: str = ...,
    ) -> None: ...
    def generate(
        self, state: str, nonce: str | None = ..., team: str | None = ...
    ) -> str: ...
