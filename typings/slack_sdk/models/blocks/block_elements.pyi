from abc import ABCMeta
from collections.abc import Sequence
from typing import Any

from slack_sdk.models.basic_objects import JsonObject

from .basic_components import (
    ConfirmObject,
    DispatchActionConfig,
    FeedbackButtonObject,
    Option,
    OptionGroup,
    PlainTextObject,
    SlackFile,
    TextObject,
    Workflow,
)

class BlockElement(JsonObject, metaclass=ABCMeta):
    attributes = ...
    logger = ...
    @property
    def subtype(self) -> str | None: ...
    def __init__(
        self, *, type: str | None = ..., subtype: str | None = ..., **others: dict
    ) -> None: ...
    @classmethod
    def parse(
        cls, block_element: dict | BlockElement
    ) -> BlockElement | TextObject | None: ...
    @classmethod
    def parse_all(
        cls, block_elements: Sequence[dict | BlockElement | TextObject]
    ) -> list[BlockElement | TextObject]: ...

class InteractiveElement(BlockElement):
    action_id_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        type: str | None = ...,
        subtype: str | None = ...,
        **others: dict,
    ) -> None: ...

class InputInteractiveElement(InteractiveElement, metaclass=ABCMeta):
    placeholder_max_length = ...
    attributes = ...
    @property
    def subtype(self) -> str | None: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        placeholder: str | TextObject | None = ...,
        type: str | None = ...,
        subtype: str | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class ButtonElement(InteractiveElement):
    type = ...
    text_max_length = ...
    url_max_length = ...
    value_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        text: str | dict | TextObject,
        action_id: str | None = ...,
        url: str | None = ...,
        value: str | None = ...,
        style: str | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        accessibility_label: str | None = ...,
        **others: dict,
    ) -> None: ...

class LinkButtonElement(ButtonElement):
    def __init__(
        self,
        *,
        text: str | dict | PlainTextObject,
        url: str,
        action_id: str | None = ...,
        style: str | None = ...,
        **others: dict,
    ) -> None: ...

class CheckboxesElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        options: Sequence[dict | Option] | None = ...,
        initial_options: Sequence[dict | Option] | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class DatePickerElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        placeholder: str | dict | TextObject | None = ...,
        initial_date: str | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class TimePickerElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        placeholder: str | dict | TextObject | None = ...,
        initial_time: str | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        timezone: str | None = ...,
        **others: dict,
    ) -> None: ...

class DateTimePickerElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        initial_date_time: int | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class FeedbackButtonsElement(InteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        positive_button: dict | FeedbackButtonObject,
        negative_button: dict | FeedbackButtonObject,
        **others: dict,
    ) -> None: ...

class ImageElement(BlockElement):
    type = ...
    image_url_max_length = ...
    alt_text_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        alt_text: str | None = ...,
        image_url: str | None = ...,
        slack_file: dict[str, Any] | SlackFile | None = ...,
        **others: dict,
    ) -> None: ...

class IconButtonElement(InteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        icon: str,
        text: str | dict | TextObject,
        accessibility_label: str | None = ...,
        value: str | None = ...,
        visible_to_user_ids: list[str] | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        **others: dict,
    ) -> None: ...

class StaticSelectElement(InputInteractiveElement):
    type = ...
    options_max_length = ...
    option_groups_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        placeholder: str | dict | TextObject | None = ...,
        action_id: str | None = ...,
        options: Sequence[dict | Option] | None = ...,
        option_groups: Sequence[dict | OptionGroup] | None = ...,
        initial_option: dict | Option | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class StaticMultiSelectElement(InputInteractiveElement):
    type = ...
    options_max_length = ...
    option_groups_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        placeholder: str | dict | TextObject | None = ...,
        action_id: str | None = ...,
        options: Sequence[Option] | None = ...,
        option_groups: Sequence[OptionGroup] | None = ...,
        initial_options: Sequence[Option] | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        max_selected_items: int | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class SelectElement(InputInteractiveElement):
    type = ...
    options_max_length = ...
    option_groups_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        placeholder: str | None = ...,
        options: Sequence[Option] | None = ...,
        option_groups: Sequence[OptionGroup] | None = ...,
        initial_option: Option | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class ExternalDataSelectElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        placeholder: str | TextObject | None = ...,
        initial_option: Option | None | OptionGroup = ...,
        min_query_length: int | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class ExternalDataMultiSelectElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        placeholder: str | dict | TextObject | None = ...,
        action_id: str | None = ...,
        min_query_length: int | None = ...,
        initial_options: Sequence[dict | Option] | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        max_selected_items: int | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class UserSelectElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        placeholder: str | dict | TextObject | None = ...,
        action_id: str | None = ...,
        initial_user: str | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class UserMultiSelectElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        placeholder: str | dict | TextObject | None = ...,
        initial_users: Sequence[str] | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        max_selected_items: int | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class ConversationFilter(JsonObject):
    attributes = ...
    logger = ...
    def __init__(
        self,
        *,
        include: Sequence[str] | None = ...,
        exclude_bot_users: bool | None = ...,
        exclude_external_shared_channels: bool | None = ...,
    ) -> None: ...
    @classmethod
    def parse(cls, filter: dict | ConversationFilter) -> ConversationFilter: ...

class ConversationSelectElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        placeholder: str | dict | TextObject | None = ...,
        action_id: str | None = ...,
        initial_conversation: str | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        response_url_enabled: bool | None = ...,
        default_to_current_conversation: bool | None = ...,
        filter: ConversationFilter | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class ConversationMultiSelectElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        placeholder: str | dict | TextObject | None = ...,
        action_id: str | None = ...,
        initial_conversations: Sequence[str] | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        max_selected_items: int | None = ...,
        default_to_current_conversation: bool | None = ...,
        filter: dict | ConversationFilter | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class ChannelSelectElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        placeholder: str | dict | TextObject | None = ...,
        action_id: str | None = ...,
        initial_channel: str | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        response_url_enabled: bool | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class ChannelMultiSelectElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        placeholder: str | dict | TextObject | None = ...,
        action_id: str | None = ...,
        initial_channels: Sequence[str] | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        max_selected_items: int | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class RichTextInputElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        placeholder: str | dict | TextObject | None = ...,
        initial_value: dict[str, Any] | RichTextBlock | None = ...,
        dispatch_action_config: dict | DispatchActionConfig | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class PlainTextInputElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        placeholder: str | dict | TextObject | None = ...,
        initial_value: str | None = ...,
        multiline: bool | None = ...,
        min_length: int | None = ...,
        max_length: int | None = ...,
        dispatch_action_config: dict | DispatchActionConfig | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class EmailInputElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        initial_value: str | None = ...,
        dispatch_action_config: dict | DispatchActionConfig | None = ...,
        focus_on_load: bool | None = ...,
        placeholder: str | dict | TextObject | None = ...,
        **others: dict,
    ) -> None: ...

class UrlInputElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        initial_value: str | None = ...,
        dispatch_action_config: dict | DispatchActionConfig | None = ...,
        focus_on_load: bool | None = ...,
        placeholder: str | dict | TextObject | None = ...,
        **others: dict,
    ) -> None: ...

class UrlSourceElement(BlockElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(self, *, url: str, text: str, **others: dict) -> None: ...

class NumberInputElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        is_decimal_allowed: bool | None = ...,
        initial_value: float | str | None = ...,
        min_value: float | str | None = ...,
        max_value: float | str | None = ...,
        dispatch_action_config: dict | DispatchActionConfig | None = ...,
        focus_on_load: bool | None = ...,
        placeholder: str | dict | TextObject | None = ...,
        **others: dict,
    ) -> None: ...

class FileInputElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        filetypes: list[str] | None = ...,
        max_files: int | None = ...,
        **others: dict,
    ) -> None: ...

class RadioButtonsElement(InputInteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        options: Sequence[dict | Option] | None = ...,
        initial_option: dict | Option | None = ...,
        confirm: dict | ConfirmObject | None = ...,
        focus_on_load: bool | None = ...,
        **others: dict,
    ) -> None: ...

class OverflowMenuElement(InteractiveElement):
    type = ...
    options_min_length = ...
    options_max_length = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        action_id: str | None = ...,
        options: Sequence[Option],
        confirm: dict | ConfirmObject | None = ...,
        **others: dict,
    ) -> None: ...

class WorkflowButtonElement(InteractiveElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        text: str | dict | TextObject,
        action_id: str | None = ...,
        workflow: dict | Workflow | None = ...,
        style: str | None = ...,
        accessibility_label: str | None = ...,
        **others: dict,
    ) -> None: ...

class RichTextElement(BlockElement): ...

class RichTextListElement(RichTextElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        elements: Sequence[dict | RichTextElement],
        style: str | None = ...,
        indent: int | None = ...,
        offset: int | None = ...,
        border: int | None = ...,
        **others: dict,
    ) -> None: ...

class RichTextPreformattedElement(RichTextElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self,
        *,
        elements: Sequence[dict | RichTextElement],
        border: int | None = ...,
        **others: dict,
    ) -> None: ...

class RichTextQuoteElement(RichTextElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self, *, elements: Sequence[dict | RichTextElement], **others: dict
    ) -> None: ...

class RichTextSectionElement(RichTextElement):
    type = ...
    @property
    def attributes(self) -> set[str]: ...
    def __init__(
        self, *, elements: Sequence[dict | RichTextElement], **others: dict
    ) -> None: ...

class RichTextElementParts:
    class TextStyle:
        def __init__(
            self,
            *,
            bold: bool | None = ...,
            italic: bool | None = ...,
            strike: bool | None = ...,
            code: bool | None = ...,
            underline: bool | None = ...,
        ) -> None: ...
        def to_dict(self, *args) -> dict: ...

    class Text(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(
            self,
            *,
            text: str,
            style: dict | RichTextElementParts.TextStyle | None = ...,
            **others: dict,
        ) -> None: ...

    class Channel(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(
            self,
            *,
            channel_id: str,
            style: dict | RichTextElementParts.TextStyle | None = ...,
            **others: dict,
        ) -> None: ...

    class User(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(
            self,
            *,
            user_id: str,
            style: dict | RichTextElementParts.TextStyle | None = ...,
            **others: dict,
        ) -> None: ...

    class Emoji(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(
            self,
            *,
            name: str,
            skin_tone: int | None = ...,
            unicode: str | None = ...,
            style: dict | RichTextElementParts.TextStyle | None = ...,
            **others: dict,
        ) -> None: ...

    class Link(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(
            self,
            *,
            url: str,
            text: str | None = ...,
            style: dict | RichTextElementParts.TextStyle | None = ...,
            **others: dict,
        ) -> None: ...

    class Team(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(
            self,
            *,
            team_id: str,
            style: dict | RichTextElementParts.TextStyle | None = ...,
            **others: dict,
        ) -> None: ...

    class UserGroup(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(
            self,
            *,
            usergroup_id: str,
            style: dict | RichTextElementParts.TextStyle | None = ...,
            **others: dict,
        ) -> None: ...

    class Date(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(
            self,
            *,
            timestamp: int,
            format: str,
            url: str | None = ...,
            fallback: str | None = ...,
            **others: dict,
        ) -> None: ...

    class Broadcast(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(self, *, range: str, **others: dict) -> None: ...

    class Color(RichTextElement):
        type = ...
        @property
        def attributes(self) -> set[str]: ...
        def __init__(self, *, value: str, **others: dict) -> None: ...
