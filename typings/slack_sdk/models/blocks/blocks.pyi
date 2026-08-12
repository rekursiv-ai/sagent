from collections.abc import Sequence
from typing import Any

from slack_sdk.models.basic_objects import JsonObject

from .basic_components import PlainTextObject, SlackFile, TextObject
from .block_elements import (
    BlockElement,
    FeedbackButtonsElement,
    IconButtonElement,
    ImageElement,
    InputInteractiveElement,
    InteractiveElement,
    RichTextElement,
    UrlSourceElement,
)

class Block(JsonObject):
    block_id_max_length = ...
    logger = ...
    @property
    def subtype(self) -> str | None: ...
    def __init__(
        self,
        *,
        type: str | None = ...,
        subtype: str | None = ...,
        block_id: str | None = ...,
    ) -> None: ...
    @classmethod
    def parse(cls, block: dict | Block) -> Block | None: ...
    @classmethod
    def parse_all(cls, blocks: Sequence[dict | Block] | None) -> list[Block]: ...

class SectionBlock(Block):
    fields_max_length = ...
    text_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        block_id: str | None = ...,
        text: str | dict | TextObject | None = ...,
        fields: Sequence[str | dict | TextObject] | None = ...,
        accessory: dict | BlockElement | None = ...,
        expand: bool | None = ...,
        **others: dict,
    ) -> None: ...

class DividerBlock(Block):
    def __init__(self, *, block_id: str | None = ..., **others: dict) -> None: ...

class ImageBlock(Block):
    @property
    def attributes(self) -> set[str]: ...

    image_url_max_length = ...
    alt_text_max_length = ...
    title_max_length = ...
    def __init__(
        self,
        *,
        alt_text: str,
        image_url: str | None = ...,
        slack_file: dict[str, Any] | SlackFile | None = ...,
        title: str | dict | PlainTextObject | None = ...,
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...

class ActionsBlock(Block):
    elements_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        elements: Sequence[dict | InteractiveElement],
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...

class ContextBlock(Block):
    elements_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        elements: Sequence[dict | ImageElement | TextObject],
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...

class ContextActionsBlock(Block):
    elements_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        elements: Sequence[dict | FeedbackButtonsElement | IconButtonElement],
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...

class InputBlock(Block):
    label_max_length = ...
    hint_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        label: str | dict | PlainTextObject,
        element: str | dict | InputInteractiveElement,
        block_id: str | None = ...,
        hint: str | dict | PlainTextObject | None = ...,
        dispatch_action: bool | None = ...,
        optional: bool | None = ...,
        **others: dict,
    ) -> None: ...

class FileBlock(Block):
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        external_id: str,
        source: str = ...,
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...

class CallBlock(Block):
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        call_id: str,
        api_decoration_available: bool | None = ...,
        call: dict[str, dict[str, Any]] | None = ...,
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...

class HeaderBlock(Block):
    text_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        block_id: str | None = ...,
        text: str | dict | TextObject | None = ...,
        **others: dict,
    ) -> None: ...

class MarkdownBlock(Block):
    text_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self, *, text: str, block_id: str | None = ..., **others: dict
    ) -> None: ...

class VideoBlock(Block):
    title_max_length = ...
    author_name_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        block_id: str | None = ...,
        alt_text: str | None = ...,
        video_url: str | None = ...,
        thumbnail_url: str | None = ...,
        title: str | dict | PlainTextObject | None = ...,
        title_url: str | None = ...,
        description: str | dict | PlainTextObject | None = ...,
        provider_icon_url: str | None = ...,
        provider_name: str | None = ...,
        author_name: str | None = ...,
        **others: dict,
    ) -> None: ...

class RichTextBlock(Block):
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        elements: Sequence[dict | RichTextElement],
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...

class TableBlock(Block):
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        rows: Sequence[Sequence[dict[str, Any]]],
        column_settings: Sequence[dict[str, Any] | None] | None = ...,
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...

class TaskCardBlock(Block):
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        task_id: str,
        title: str,
        details: RichTextBlock | dict | None = ...,
        output: RichTextBlock | dict | None = ...,
        sources: Sequence[UrlSourceElement | dict] | None = ...,
        status: str,
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...

class PlanBlock(Block):
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        title: str,
        tasks: Sequence[dict | TaskCardBlock] | None = ...,
        block_id: str | None = ...,
        **others: dict,
    ) -> None: ...
