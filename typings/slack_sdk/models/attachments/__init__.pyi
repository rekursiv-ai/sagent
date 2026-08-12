from abc import ABCMeta, abstractmethod
from collections.abc import Sequence

from slack_sdk.models.basic_objects import EnumValidator, JsonObject, JsonValidator
from slack_sdk.models.blocks import Block, ButtonStyles, ConfirmObject, Option

class Action(JsonObject):
    def __init__(
        self, *, text: str, subtype: str, name: str | None = ..., url: str | None = ...
    ) -> None: ...
    @JsonValidator("name or url attribute is required")
    def name_or_url_present(self) -> bool: ...
    def to_dict(self) -> dict: ...

class ActionButton(Action):
    @property
    def attributes(self) -> set[str]: ...

    value_max_length = ...
    def __init__(
        self,
        *,
        name: str,
        text: str,
        value: str,
        confirm: ConfirmObject | None = ...,
        style: str | None = ...,
    ) -> None: ...
    @JsonValidator(...)
    def value_length(self) -> bool: ...
    @EnumValidator("style", ButtonStyles)
    def style_valid(self) -> bool: ...
    def to_dict(self) -> dict: ...

class ActionLinkButton(Action):
    def __init__(self, *, text: str, url: str) -> None: ...

class AbstractActionSelector(Action, metaclass=ABCMeta):
    DataSourceTypes = ...
    @property
    @abstractmethod
    def data_source(self) -> str: ...
    def __init__(
        self, *, name: str, text: str, selected_option: Option | None = ...
    ) -> None: ...
    @EnumValidator("data_source", DataSourceTypes)
    def data_source_valid(self) -> bool: ...
    def to_dict(self) -> dict: ...

class ActionUserSelector(AbstractActionSelector):
    def __init__(
        self, name: str, text: str, selected_user: Option | None = ...
    ) -> None: ...

class ActionChannelSelector(AbstractActionSelector):
    def __init__(
        self, name: str, text: str, selected_channel: Option | None = ...
    ) -> None: ...

class ActionConversationSelector(AbstractActionSelector):
    def __init__(
        self, name: str, text: str, selected_conversation: Option | None = ...
    ) -> None: ...

class ActionExternalSelector(AbstractActionSelector):
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        name: str,
        text: str,
        selected_option: Option | None = ...,
        min_query_length: int | None = ...,
    ) -> None: ...

SeededColors = ...

class AttachmentField(JsonObject):
    def __init__(
        self, *, title: str | None = ..., value: str | None = ..., short: bool = ...
    ) -> None: ...

class Attachment(JsonObject):
    fields: Sequence[AttachmentField]
    MarkdownFields = ...
    footer_max_length = ...
    def __init__(
        self,
        *,
        text: str,
        fallback: str | None = ...,
        fields: Sequence[AttachmentField] | None = ...,
        color: str | None = ...,
        markdown_in: Sequence[str] | None = ...,
        title: str | None = ...,
        title_link: str | None = ...,
        pretext: str | None = ...,
        author_name: str | None = ...,
        author_subname: str | None = ...,
        author_link: str | None = ...,
        author_icon: str | None = ...,
        image_url: str | None = ...,
        thumb_url: str | None = ...,
        footer: str | None = ...,
        footer_icon: str | None = ...,
        ts: int | None = ...,
    ) -> None: ...
    @JsonValidator(...)
    def footer_length(self) -> bool: ...
    @JsonValidator(...)
    def ts_without_footer(self) -> bool: ...
    @EnumValidator("markdown_in", MarkdownFields)
    def markdown_in_valid(self) -> bool: ...
    @JsonValidator(...)
    def color_valid(self) -> bool: ...
    @JsonValidator(...)
    def image_url_and_thumb_url_populated(self) -> bool: ...
    @JsonValidator("name must be present if link is present")
    def author_link_without_author_name(self) -> bool: ...
    @JsonValidator("icon must be present if link is present")
    def author_link_without_author_icon(self) -> bool: ...
    def to_dict(self) -> dict: ...

class BlockAttachment(Attachment):
    blocks: list[Block]
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        blocks: Sequence[Block],
        color: str | None = ...,
        fallback: str | None = ...,
    ) -> None: ...
    @JsonValidator(...)
    def fields_attribute_absent(self) -> bool: ...
    def to_dict(self) -> dict: ...

class InteractiveAttachment(Attachment):
    @property
    def attributes(self) -> set[str]: ...

    actions_max_length = ...
    def __init__(
        self,
        *,
        actions: Sequence[Action],
        callback_id: str,
        text: str,
        fallback: str | None = ...,
        fields: Sequence[AttachmentField] | None = ...,
        color: str | None = ...,
        markdown_in: Sequence[str] | None = ...,
        title: str | None = ...,
        title_link: str | None = ...,
        pretext: str | None = ...,
        author_name: str | None = ...,
        author_subname: str | None = ...,
        author_link: str | None = ...,
        author_icon: str | None = ...,
        image_url: str | None = ...,
        thumb_url: str | None = ...,
        footer: str | None = ...,
        footer_icon: str | None = ...,
        ts: int | None = ...,
    ) -> None: ...
    @JsonValidator(...)
    def actions_length(self) -> bool: ...
    def to_dict(self) -> dict: ...
