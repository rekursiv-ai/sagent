from typing import Any

from slack_sdk.models.basic_objects import EnumValidator, JsonObject

class Metadata(JsonObject):
    attributes = ...
    def __init__(
        self, event_type: str, event_payload: dict[str, Any], **kwargs
    ) -> None: ...

EntityType = ...
CustomFieldType = ...

class ExternalRef(JsonObject):
    attributes = ...
    def __init__(self, id: str, type: str | None = ..., **kwargs) -> None: ...

class FileEntitySlackFile(JsonObject):
    attributes = ...
    def __init__(self, id: str, type: str | None = ..., **kwargs) -> None: ...

class EntityIconSlackFile(JsonObject):
    attributes = ...
    def __init__(
        self, id: str | None = ..., url: str | None = ..., **kwargs
    ) -> None: ...

class EntityIconField(JsonObject):
    attributes = ...
    def __init__(
        self,
        alt_text: str,
        url: str | None = ...,
        slack_file: dict[str, Any] | EntityIconSlackFile | None = ...,
        **kwargs,
    ) -> None: ...

class EntityEditSelectConfig(JsonObject):
    attributes = ...
    def __init__(
        self,
        current_value: str | None = ...,
        current_values: list[str] | None = ...,
        static_options: list[dict[str, Any]] | None = ...,
        fetch_options_dynamically: bool | None = ...,
        min_query_length: int | None = ...,
        **kwargs,
    ) -> None: ...

class EntityEditNumberConfig(JsonObject):
    attributes = ...
    def __init__(
        self,
        is_decimal_allowed: bool | None = ...,
        min_value: float | None = ...,
        max_value: float | None = ...,
        **kwargs,
    ) -> None: ...

class EntityEditTextConfig(JsonObject):
    attributes = ...
    def __init__(
        self, min_length: int | None = ..., max_length: int | None = ..., **kwargs
    ) -> None: ...

class EntityEditSupport(JsonObject):
    attributes = ...
    def __init__(
        self,
        enabled: bool,
        placeholder: dict[str, Any] | None = ...,
        hint: dict[str, Any] | None = ...,
        optional: bool | None = ...,
        select: dict[str, Any] | EntityEditSelectConfig | None = ...,
        number: dict[str, Any] | EntityEditNumberConfig | None = ...,
        text: dict[str, Any] | EntityEditTextConfig | None = ...,
        **kwargs,
    ) -> None: ...

class EntityFullSizePreviewError(JsonObject):
    attributes = ...
    def __init__(self, code: str, message: str | None = ..., **kwargs) -> None: ...

class EntityFullSizePreview(JsonObject):
    attributes = ...
    def __init__(
        self,
        is_supported: bool,
        preview_url: str | None = ...,
        mime_type: str | None = ...,
        error: dict[str, Any] | EntityFullSizePreviewError | None = ...,
        **kwargs,
    ) -> None: ...

class EntityUserIDField(JsonObject):
    attributes = ...
    def __init__(self, user_id: str, **kwargs) -> None: ...

class EntityUserField(JsonObject):
    attributes = ...
    def __init__(
        self,
        text: str,
        url: str | None = ...,
        email: str | None = ...,
        icon: dict[str, Any] | EntityIconField | None = ...,
        **kwargs,
    ) -> None: ...

class EntityRefField(JsonObject):
    attributes = ...
    def __init__(
        self,
        entity_url: str,
        external_ref: dict[str, Any] | ExternalRef,
        title: str,
        display_type: str | None = ...,
        icon: dict[str, Any] | EntityIconField | None = ...,
        **kwargs,
    ) -> None: ...

class EntityTypedField(JsonObject):
    attributes = ...
    def __init__(
        self,
        type: str,
        label: str | None = ...,
        value: str | int | None = ...,
        link: str | None = ...,
        icon: dict[str, Any] | EntityIconField | None = ...,
        long: bool | None = ...,
        format: str | None = ...,
        image_url: str | None = ...,
        slack_file: dict[str, Any] | None = ...,
        alt_text: str | None = ...,
        edit: dict[str, Any] | EntityEditSupport | None = ...,
        tag_color: str | None = ...,
        user: dict[str, Any] | EntityUserIDField | EntityUserField | None = ...,
        entity_ref: dict[str, Any] | EntityRefField | None = ...,
        **kwargs,
    ) -> None: ...

class EntityStringField(JsonObject):
    attributes = ...
    def __init__(
        self,
        value: str,
        label: str | None = ...,
        format: str | None = ...,
        link: str | None = ...,
        icon: dict[str, Any] | EntityIconField | None = ...,
        long: bool | None = ...,
        type: str | None = ...,
        tag_color: str | None = ...,
        edit: dict[str, Any] | EntityEditSupport | None = ...,
        **kwargs,
    ) -> None: ...

class EntityTimestampField(JsonObject):
    attributes = ...
    def __init__(
        self,
        value: int,
        label: str | None = ...,
        type: str | None = ...,
        edit: dict[str, Any] | EntityEditSupport | None = ...,
        **kwargs,
    ) -> None: ...

class EntityImageField(JsonObject):
    attributes = ...
    def __init__(
        self,
        alt_text: str,
        label: str | None = ...,
        image_url: str | None = ...,
        slack_file: dict[str, Any] | None = ...,
        title: str | None = ...,
        type: str | None = ...,
        **kwargs,
    ) -> None: ...

class EntityBooleanCheckboxField(JsonObject):
    attributes = ...
    def __init__(
        self, type: str, text: str, description: str | None, **kwargs
    ) -> None: ...

class EntityBooleanTextField(JsonObject):
    attributes = ...
    def __init__(
        self,
        type: str,
        true_text: str,
        false_text: str,
        true_description: str | None,
        false_description: str | None,
        **kwargs,
    ) -> None: ...

class EntityArrayItemField(JsonObject):
    attributes = ...
    def __init__(
        self,
        type: str | None = ...,
        label: str | None = ...,
        value: str | int | None = ...,
        link: str | None = ...,
        icon: dict[str, Any] | EntityIconField | None = ...,
        long: bool | None = ...,
        format: str | None = ...,
        image_url: str | None = ...,
        slack_file: dict[str, Any] | None = ...,
        alt_text: str | None = ...,
        edit: dict[str, Any] | EntityEditSupport | None = ...,
        tag_color: str | None = ...,
        user: dict[str, Any] | EntityUserIDField | EntityUserField | None = ...,
        entity_ref: dict[str, Any] | EntityRefField | None = ...,
        **kwargs,
    ) -> None: ...

class EntityCustomField(JsonObject):
    attributes = ...
    def __init__(
        self,
        label: str,
        key: str,
        type: str,
        value: str | int | list[dict[str, Any] | EntityArrayItemField] | None = ...,
        link: str | None = ...,
        icon: dict[str, Any] | EntityIconField | None = ...,
        long: bool | None = ...,
        format: str | None = ...,
        image_url: str | None = ...,
        slack_file: dict[str, Any] | None = ...,
        alt_text: str | None = ...,
        tag_color: str | None = ...,
        edit: dict[str, Any] | EntityEditSupport | None = ...,
        item_type: str | None = ...,
        user: dict[str, Any] | EntityUserIDField | EntityUserField | None = ...,
        entity_ref: dict[str, Any] | EntityRefField | None = ...,
        boolean: dict[str, Any]
        | EntityBooleanCheckboxField
        | EntityBooleanTextField
        | None = ...,
        **kwargs,
    ) -> None: ...
    @EnumValidator("type", CustomFieldType)
    def type_valid(self) -> bool: ...

class FileEntityFields(JsonObject):
    attributes = ...
    def __init__(
        self,
        preview: dict[str, Any] | EntityImageField | None = ...,
        created_by: dict[str, Any] | EntityTypedField | None = ...,
        date_created: dict[str, Any] | EntityTimestampField | None = ...,
        date_updated: dict[str, Any] | EntityTimestampField | None = ...,
        last_modified_by: dict[str, Any] | EntityTypedField | None = ...,
        file_size: dict[str, Any] | EntityStringField | None = ...,
        mime_type: dict[str, Any] | EntityStringField | None = ...,
        full_size_preview: dict[str, Any] | EntityFullSizePreview | None = ...,
        **kwargs,
    ) -> None: ...

class TaskEntityFields(JsonObject):
    attributes = ...
    def __init__(
        self,
        description: dict[str, Any] | EntityStringField | None = ...,
        created_by: dict[str, Any] | EntityTypedField | None = ...,
        date_created: dict[str, Any] | EntityTimestampField | None = ...,
        date_updated: dict[str, Any] | EntityTimestampField | None = ...,
        assignee: dict[str, Any] | EntityTypedField | None = ...,
        status: dict[str, Any] | EntityStringField | None = ...,
        due_date: dict[str, Any] | EntityTypedField | None = ...,
        priority: dict[str, Any] | EntityStringField | None = ...,
        **kwargs,
    ) -> None: ...

class IncidentEntityFields(JsonObject):
    attributes = ...
    def __init__(
        self,
        status: dict[str, Any] | EntityStringField | None = ...,
        priority: dict[str, Any] | EntityStringField | None = ...,
        urgency: dict[str, Any] | EntityStringField | None = ...,
        created_by: dict[str, Any] | EntityTypedField | None = ...,
        assigned_to: dict[str, Any] | EntityTypedField | None = ...,
        date_created: dict[str, Any] | EntityTimestampField | None = ...,
        date_updated: dict[str, Any] | EntityTimestampField | None = ...,
        description: dict[str, Any] | EntityStringField | None = ...,
        service: dict[str, Any] | EntityStringField | None = ...,
        **kwargs,
    ) -> None: ...

class ContentItemEntityFields(JsonObject):
    attributes = ...
    def __init__(
        self,
        preview: dict[str, Any] | EntityImageField | None = ...,
        description: dict[str, Any] | EntityStringField | None = ...,
        created_by: dict[str, Any] | EntityTypedField | None = ...,
        date_created: dict[str, Any] | EntityTimestampField | None = ...,
        date_updated: dict[str, Any] | EntityTimestampField | None = ...,
        last_modified_by: dict[str, Any] | EntityTypedField | None = ...,
        **kwargs,
    ) -> None: ...

class EntityActionProcessingState(JsonObject):
    attributes = ...
    def __init__(
        self, enabled: bool, interstitial_text: str | None = ..., **kwargs
    ) -> None: ...

class EntityActionButton(JsonObject):
    attributes = ...
    def __init__(
        self,
        text: str,
        action_id: str,
        value: str | None = ...,
        style: str | None = ...,
        url: str | None = ...,
        accessibility_label: str | None = ...,
        processing_state: dict[str, Any] | EntityActionProcessingState | None = ...,
        **kwargs,
    ) -> None: ...

class EntityTitle(JsonObject):
    attributes = ...
    def __init__(
        self, text: str, edit: dict[str, Any] | EntityEditSupport | None = ..., **kwargs
    ) -> None: ...

class EntityAttributes(JsonObject):
    attributes = ...
    def __init__(
        self,
        title: dict[str, Any] | EntityTitle,
        display_type: str | None = ...,
        display_id: str | None = ...,
        product_icon: dict[str, Any] | EntityIconField | None = ...,
        product_name: str | None = ...,
        locale: str | None = ...,
        full_size_preview: dict[str, Any] | EntityFullSizePreview | None = ...,
        metadata_last_modified: int | None = ...,
        **kwargs,
    ) -> None: ...

class EntityActions(JsonObject):
    attributes = ...
    def __init__(
        self,
        primary_actions: list[dict[str, Any] | EntityActionButton] | None = ...,
        overflow_actions: list[dict[str, Any] | EntityActionButton] | None = ...,
        **kwargs,
    ) -> None: ...

class EntityPayload(JsonObject):
    attributes = ...
    def __init__(
        self,
        attributes: dict[str, Any] | EntityAttributes,
        fields: dict[str, Any]
        | ContentItemEntityFields
        | FileEntityFields
        | IncidentEntityFields
        | TaskEntityFields
        | None = ...,
        custom_fields: list[dict[str, Any] | EntityCustomField] | None = ...,
        slack_file: dict[str, Any] | FileEntitySlackFile | None = ...,
        display_order: list[str] | None = ...,
        actions: dict[str, Any] | EntityActions | None = ...,
        **kwargs,
    ) -> None: ...
    @property
    def entity_attributes(self) -> dict[str, Any] | EntityAttributes: ...
    @entity_attributes.setter
    def entity_attributes(self, value: dict[str, Any] | EntityAttributes) -> None: ...
    def get_object_attribute(
        self, key: str
    ) -> dict[str, Any] | EntityAttributes | Any | None: ...

class EntityMetadata(JsonObject):
    attributes = ...
    def __init__(
        self,
        entity_type: str,
        entity_payload: dict[str, Any] | EntityPayload,
        external_ref: dict[str, Any] | ExternalRef,
        url: str,
        app_unfurl_url: str | None = ...,
        **kwargs,
    ) -> None: ...
    @EnumValidator("entity_type", EntityType)
    def entity_type_valid(self) -> bool: ...

class EventAndEntityMetadata(JsonObject):
    attributes = ...
    def __init__(
        self,
        event_type: str | None = ...,
        event_payload: dict[str, Any] | None = ...,
        entities: list[dict[str, Any] | EntityMetadata] | None = ...,
        **kwargs,
    ) -> None: ...
