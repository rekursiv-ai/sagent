from collections.abc import Sequence

from slack_sdk.models.basic_objects import JsonObject
from slack_sdk.models.blocks import Block, Option, PlainTextObject

class View(JsonObject):
    types = ...
    attributes = ...
    def __init__(
        self,
        type: str,
        id: str | None = ...,
        callback_id: str | None = ...,
        external_id: str | None = ...,
        team_id: str | None = ...,
        bot_id: str | None = ...,
        app_id: str | None = ...,
        root_view_id: str | None = ...,
        previous_view_id: str | None = ...,
        title: str | dict | PlainTextObject | None = ...,
        submit: str | dict | PlainTextObject | None = ...,
        close: str | dict | PlainTextObject | None = ...,
        blocks: Sequence[dict | Block] | None = ...,
        private_metadata: str | None = ...,
        state: dict | ViewState | None = ...,
        hash: str | None = ...,
        clear_on_close: bool | None = ...,
        notify_on_close: bool | None = ...,
        **kwargs,
    ) -> None: ...

    title_max_length = ...
    blocks_max_length = ...
    close_max_length = ...
    submit_max_length = ...
    private_metadata_max_length = ...
    callback_id_max_length: int = ...

class ViewState(JsonObject):
    attributes = ...
    logger = ...
    def __init__(
        self, *, values: dict[str, dict[str, dict | ViewStateValue]]
    ) -> None: ...
    def to_dict(self, *args) -> dict[str, dict[str, dict[str, dict]]]: ...

class ViewStateValue(JsonObject):
    attributes = ...
    def __init__(
        self,
        *,
        type: str | None = ...,
        value: str | None = ...,
        selected_date: str | None = ...,
        selected_time: str | None = ...,
        selected_conversation: str | None = ...,
        selected_channel: str | None = ...,
        selected_user: str | None = ...,
        selected_option: dict | Option | None = ...,
        selected_conversations: Sequence[str] | None = ...,
        selected_channels: Sequence[str] | None = ...,
        selected_users: Sequence[str] | None = ...,
        selected_options: Sequence[dict | Option] | None = ...,
    ) -> None: ...
