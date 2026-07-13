class RedirectUriPageRenderer:
    def __init__(
        self,
        *,
        install_path: str,
        redirect_uri_path: str,
        success_url: str | None = ...,
        failure_url: str | None = ...,
    ) -> None: ...
    def render_success_page(
        self,
        app_id: str,
        team_id: str | None,
        is_enterprise_install: bool | None = ...,
        enterprise_url: str | None = ...,
    ) -> str: ...
    def render_failure_page(self, reason: str) -> str: ...
