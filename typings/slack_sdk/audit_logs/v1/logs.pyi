from typing import Any

class App:
    id: str | None
    name: str | None
    is_distributed: bool | None
    is_directory_approved: bool | None
    is_workflow_app: bool | None
    scopes: list[str] | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        name: str | None = ...,
        is_distributed: bool | None = ...,
        is_directory_approved: bool | None = ...,
        is_workflow_app: bool | None = ...,
        scopes: list[str] | None = ...,
        **kwargs,
    ) -> None: ...

class User:
    id: str | None
    name: str | None
    email: str | None
    team: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        name: str | None = ...,
        email: str | None = ...,
        team: str | None = ...,
        **kwargs,
    ) -> None: ...

class Actor:
    type: str | None
    user: User | None
    unknown_fields: dict[str, Any]
    def __init__(
        self, type: str | None = ..., user: User | dict[str, Any] | None = ..., **kwargs
    ) -> None: ...

class Location:
    type: str | None
    id: str | None
    name: str | None
    domain: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        type: str | None = ...,
        id: str | None = ...,
        name: str | None = ...,
        domain: str | None = ...,
        **kwargs,
    ) -> None: ...

class Context:
    location: Location | None
    ua: str | None
    ip_address: str | None
    session_id: str | None
    app: App | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        location: Location | dict[str, Any] | None = ...,
        ua: str | None = ...,
        ip_address: str | None = ...,
        session_id: str | None = ...,
        app: App | dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...

class RetentionPolicy:
    type: str | None
    duration_days: int | None
    def __init__(
        self, *, type: str | None = ..., duration_days: int | None = ..., **kwargs
    ) -> None: ...

class ConversationPref:
    type: list[str] | None
    user: list[str] | None
    def __init__(
        self, *, type: list[str] | None = ..., user: list[str] | None = ..., **kwargs
    ) -> None: ...

class FeatureEnablement:
    enabled: bool | None
    def __init__(self, *, enabled: bool | None = ..., **kwargs) -> None: ...

class SharedWith:
    channel_id: str | None
    access_level: str | None
    def __init__(
        self, *, channel_id: str | None = ..., access_level: str | None = ..., **kwargs
    ) -> None: ...

class Profile:
    real_name: str | None
    first_name: str | None
    last_name: str | None
    display_name: str | None
    image_original: str | None
    image_24: str | None
    image_32: str | None
    image_48: str | None
    image_72: str | None
    image_192: str | None
    image_512: str | None
    image_1024: str | None
    def __init__(
        self,
        *,
        real_name: str | None = ...,
        first_name: str | None = ...,
        last_name: str | None = ...,
        display_name: str | None = ...,
        image_original: str | None = ...,
        image_24: str | None = ...,
        image_32: str | None = ...,
        image_48: str | None = ...,
        image_72: str | None = ...,
        image_192: str | None = ...,
        image_512: str | None = ...,
        image_1024: str | None = ...,
        **kwargs,
    ) -> None: ...

class SpaceFileId:
    payload: str | None
    def __init__(self, *, payload: str | None = ..., **kwargs) -> None: ...

class AttributeItems:
    type: str | None
    def __init__(self, *, type: str | None = ..., **kwargs) -> None: ...

class Attribute:
    name: str | None
    type: str | None
    items: AttributeItems | None
    def __init__(
        self,
        *,
        name: str | None = ...,
        type: str | None = ...,
        items: AttributeItems | None = ...,
        **kwargs,
    ) -> None: ...

class AAARuleActionResolution:
    value: str | None
    def __init__(self, *, value: str | None = ..., **kwargs) -> None: ...

class AAARuleActionNotify:
    entity_type: str | None
    def __init__(self, *, entity_type: str | None = ..., **kwargs) -> None: ...

class AAARuleAction:
    resolution: AAARuleActionResolution | None
    notify: list[AAARuleActionNotify] | None
    def __init__(
        self,
        *,
        resolution: dict[str, Any] | AAARuleActionResolution | None = ...,
        notify: list[dict[str, Any] | AAARuleActionNotify] | None = ...,
        **kwargs,
    ) -> None: ...

class AAARuleConditionValue:
    field: str | None
    values: list[str] | None
    datatype: str | None
    operator: str | None
    def __init__(
        self,
        *,
        field: str | None = ...,
        values: list[str] | None = ...,
        datatype: str | None = ...,
        operator: str | None = ...,
        **kwargs,
    ) -> None: ...

class AAARuleCondition:
    datatype: str | None
    operator: str | None
    values: list[AAARuleConditionValue] | None
    entity_type: str | None
    def __init__(
        self,
        *,
        datatype: str | None = ...,
        operator: str | None = ...,
        values: list[dict[str, Any] | AAARuleConditionValue] | None = ...,
        entity_type: str | None = ...,
        **kwargs,
    ) -> None: ...

class AAARule:
    id: str | None
    team_id: str | None
    title: str | None
    action: AAARuleAction | None
    condition: AAARuleCondition | None
    def __init__(
        self,
        *,
        id: str | None = ...,
        team_id: str | None = ...,
        title: str | None = ...,
        action: dict[str, Any] | AAARuleAction | None = ...,
        condition: dict[str, Any] | AAARuleCondition | None = ...,
        **kwargs,
    ) -> None: ...

class AAARequest:
    id: str | None
    team_id: str | None
    def __init__(
        self, *, id: str | None = ..., team_id: str | None = ..., **kwargs
    ) -> None: ...

class Details:
    name: str | None
    new_value: str | list[str] | dict[str, Any] | None
    previous_value: str | list[str] | dict[str, Any] | None
    expires_on: int | None
    mobile_only: bool | None
    web_only: bool | None
    non_sso_only: bool | None
    type: str | None
    is_workflow: bool | None
    inviter: User | None
    kicker: User | None
    shared_to: str | None
    reason: str | None
    origin_team: str | None
    target_team: str | None
    is_internal_integration: bool | None
    cleared_resolution: str | None
    app_owner_id: str | None
    bot_scopes: list[str] | None
    new_scopes: list[str] | None
    previous_scopes: list[str] | None
    granular_bot_token: bool | None
    scopes: list[str] | None
    scopes_bot: list[str] | None
    resolution: str | None
    app_previously_resolved: bool | None
    admin_app_id: str | None
    bot_id: str | None
    installer_user_id: str | None
    approver_id: str | None
    approval_type: str | None
    app_previously_approved: bool | None
    old_scopes: list[str] | None
    channels: list[str] | None
    permissions: list[dict[str, Any]] | None
    new_version_id: str | None
    trigger: str | None
    export_type: str | None
    export_start_ts: str | None
    export_end_ts: str | None
    barrier_id: str | None
    primary_usergroup_id: str | None
    barriered_from_usergroup_ids: list[str] | None
    restricted_subjects: list[str] | None
    duration: int | None
    desktop_app_browser_quit: bool | None
    invite_id: str | None
    external_organization_id: str | None
    external_organization_name: str | None
    external_user_id: str | None
    external_user_email: str | None
    channel_id: str | None
    added_team_id: str | None
    unknown_fields: dict[str, Any]
    is_token_rotation_enabled_app: bool | None
    old_retention_policy: RetentionPolicy | None
    new_retention_policy: RetentionPolicy | None
    who_can_post: ConversationPref | None
    can_thread: ConversationPref | None
    is_external_limited: bool | None
    exporting_team_id: int | None
    session_search_start: int | None
    deprecation_search_end: int | None
    is_error: bool | None
    creator: str | None
    team: str | None
    app_id: str | None
    enable_at_here: FeatureEnablement | None
    enable_at_channel: FeatureEnablement | None
    can_huddle: FeatureEnablement | None
    url_private: str | None
    shared_with: SharedWith | None
    initiated_by: str | None
    source_team: str | None
    destination_team: str | None
    succeeded_users: list[str] | None
    failed_users: list[str] | None
    enterprise: str | None
    subteam: str | None
    action: str | None
    idp_group_member_count: int | None
    workspace_member_count: int | None
    added_user_count: int | None
    added_user_error_count: int | None
    reactivated_user_count: int | None
    removed_user_count: int | None
    removed_user_error_count: int | None
    total_removal_count: int | None
    is_flagged: str | None
    target_user: str | None
    idp_config_id: str | None
    config_type: str | None
    idp_entity_id_hash: str | None
    label: str | None
    previous_profile: Profile | None
    new_profile: Profile | None
    target_user_id: str | None
    space_file_id: SpaceFileId | None
    target_entity: str | None
    target_entity_id: str | None
    changed_permissions: list[str] | None
    datastore_name: str | None
    attributes: list[Attribute] | None
    channel: str | None
    entity_type: str | None
    actor: str | None
    access_level: str | None
    functions: list[str] | None
    workflows: list[str] | None
    datastores: list[str] | None
    permissions_updated: bool | None
    matched_rule: AAARule | None
    request: AAARequest | None
    rules_checked: list[AAARule] | None
    disconnecting_team: str | None
    is_channel_canvas: bool | None
    linked_channel_id: str | None
    column_id: str | None
    row_id: str | None
    cell_date_updated: int | None
    view_id: str | None
    user: str | None
    def __init__(
        self,
        *,
        name: str | None = ...,
        new_value: str | list[str] | dict[str, Any] | None = ...,
        previous_value: str | list[str] | dict[str, Any] | None = ...,
        expires_on: int | None = ...,
        mobile_only: bool | None = ...,
        web_only: bool | None = ...,
        non_sso_only: bool | None = ...,
        type: str | None = ...,
        is_workflow: bool | None = ...,
        inviter: dict[str, Any] | User | None = ...,
        kicker: dict[str, Any] | User | None = ...,
        shared_to: str | None = ...,
        reason: str | None = ...,
        origin_team: str | None = ...,
        target_team: str | None = ...,
        is_internal_integration: bool | None = ...,
        cleared_resolution: str | None = ...,
        app_owner_id: str | None = ...,
        bot_scopes: list[str] | None = ...,
        new_scopes: list[str] | None = ...,
        previous_scopes: list[str] | None = ...,
        granular_bot_token: bool | None = ...,
        scopes: list[str] | None = ...,
        scopes_bot: list[str] | None = ...,
        resolution: str | None = ...,
        app_previously_resolved: bool | None = ...,
        admin_app_id: str | None = ...,
        bot_id: str | None = ...,
        installer_user_id: str | None = ...,
        approver_id: str | None = ...,
        approval_type: str | None = ...,
        app_previously_approved: bool | None = ...,
        old_scopes: list[str] | None = ...,
        channels: list[str] | None = ...,
        permissions: list[dict[str, Any]] | None = ...,
        new_version_id: str | None = ...,
        trigger: str | None = ...,
        export_type: str | None = ...,
        export_start_ts: str | None = ...,
        export_end_ts: str | None = ...,
        barrier_id: str | None = ...,
        primary_usergroup_id: str | None = ...,
        barriered_from_usergroup_ids: list[str] | None = ...,
        restricted_subjects: list[str] | None = ...,
        duration: int | None = ...,
        desktop_app_browser_quit: bool | None = ...,
        invite_id: str | None = ...,
        external_organization_id: str | None = ...,
        external_organization_name: str | None = ...,
        external_user_id: str | None = ...,
        external_user_email: str | None = ...,
        channel_id: str | None = ...,
        added_team_id: str | None = ...,
        is_token_rotation_enabled_app: bool | None = ...,
        old_retention_policy: dict[str, Any] | RetentionPolicy | None = ...,
        new_retention_policy: dict[str, Any] | RetentionPolicy | None = ...,
        who_can_post: dict[str, list[str]] | ConversationPref | None = ...,
        can_thread: dict[str, list[str]] | ConversationPref | None = ...,
        is_external_limited: bool | None = ...,
        exporting_team_id: int | None = ...,
        session_search_start: int | None = ...,
        deprecation_search_end: int | None = ...,
        is_error: bool | None = ...,
        creator: str | None = ...,
        team: str | None = ...,
        app_id: str | None = ...,
        enable_at_here: dict[str, Any] | FeatureEnablement | None = ...,
        enable_at_channel: dict[str, Any] | FeatureEnablement | None = ...,
        can_huddle: dict[str, Any] | FeatureEnablement | None = ...,
        url_private: str | None = ...,
        shared_with: dict[str, Any] | SharedWith | None = ...,
        initiated_by: str | None = ...,
        source_team: str | None = ...,
        destination_team: str | None = ...,
        succeeded_users: list[str] | str | None = ...,
        failed_users: list[str] | str | None = ...,
        enterprise: str | None = ...,
        subteam: str | None = ...,
        action: str | None = ...,
        idp_group_member_count: int | None = ...,
        workspace_member_count: int | None = ...,
        added_user_count: int | None = ...,
        added_user_error_count: int | None = ...,
        reactivated_user_count: int | None = ...,
        removed_user_count: int | None = ...,
        removed_user_error_count: int | None = ...,
        total_removal_count: int | None = ...,
        is_flagged: str | None = ...,
        target_user: str | None = ...,
        idp_config_id: str | None = ...,
        config_type: str | None = ...,
        idp_entity_id_hash: str | None = ...,
        label: str | None = ...,
        previous_profile: dict[str, Any] | Profile | None = ...,
        new_profile: dict[str, Any] | Profile | None = ...,
        target_user_id: str | None = ...,
        space_file_id: dict[str, Any] | SpaceFileId | None = ...,
        target_entity: str | None = ...,
        target_entity_id: str | None = ...,
        changed_permissions: list[str] | None = ...,
        datastore_name: str | None = ...,
        attributes: list[dict[str, str] | Attribute] | None = ...,
        channel: str | None = ...,
        entity_type: str | None = ...,
        actor: str | None = ...,
        access_level: str | None = ...,
        functions: list[str] | None = ...,
        workflows: list[str] | None = ...,
        datastores: list[str] | None = ...,
        permissions_updated: bool | None = ...,
        matched_rule: dict[str, Any] | AAARule | None = ...,
        request: dict[str, Any] | AAARequest | None = ...,
        rules_checked: list[dict[str, Any] | AAARule] | None = ...,
        disconnecting_team: str | None = ...,
        is_channel_canvas: bool | None = ...,
        linked_channel_id: str | None = ...,
        column_id: str | None = ...,
        row_id: str | None = ...,
        cell_date_updated: int | None = ...,
        view_id: str | None = ...,
        user: str | None = ...,
        **kwargs,
    ) -> None: ...

class Channel:
    id: str | None
    privacy: str | None
    name: str | None
    is_shared: bool | None
    is_org_shared: bool | None
    teams_shared_with: list[str] | None
    original_connected_channel_id: str | None
    is_salesforce_channel: bool | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        privacy: str | None = ...,
        name: str | None = ...,
        is_shared: bool | None = ...,
        is_org_shared: bool | None = ...,
        teams_shared_with: list[str] | None = ...,
        original_connected_channel_id: str | None = ...,
        is_salesforce_channel: bool | None = ...,
        **kwargs,
    ) -> None: ...

class File:
    id: str | None
    name: str | None
    filetype: str | None
    title: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        name: str | None = ...,
        filetype: str | None = ...,
        title: str | None = ...,
        **kwargs,
    ) -> None: ...

class Usergroup:
    id: str | None
    name: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self, *, id: str | None = ..., name: str | None = ..., **kwargs
    ) -> None: ...

class Message:
    channel: str | None
    team: str | None
    timestamp: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        channel: str | None = ...,
        team: str | None = ...,
        timestamp: str | None = ...,
        **kwargs,
    ) -> None: ...

class Huddle:
    id: str | None
    date_start: int | None
    date_end: int | None
    participants: list[str] | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        date_start: int | None = ...,
        date_end: int | None = ...,
        participants: list[str] | None = ...,
        **kwargs,
    ) -> None: ...

class Role:
    id: str | None
    name: str | None
    type: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        name: str | None = ...,
        type: str | None = ...,
        **kwargs,
    ) -> None: ...

class Workflow:
    id: str | None
    name: str | None
    domain: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        name: str | None = ...,
        domain: str | None = ...,
        **kwargs,
    ) -> None: ...

class InformationBarrier:
    id: str | None
    primary_usergroup: str | None
    barriered_from_usergroups: list[str] | None
    restricted_subjects: list[str] | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        primary_usergroup: str | None = ...,
        barriered_from_usergroups: list[str] | None = ...,
        restricted_subjects: list[str] | None = ...,
        **kwargs,
    ) -> None: ...

class WorkflowV2StepConfiguration:
    name: str | None
    step_function_type: str | None
    step_function_app_id: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        name: str | None = ...,
        step_function_type: str | None = ...,
        step_function_app_id: str | None = ...,
        **kwargs,
    ) -> None: ...

class WorkflowV2:
    id: str | None
    app_id: str | None
    date_updated: int | None
    callback_id: str | None
    name: str | None
    updated_by: str | None
    step_configuration: list[WorkflowV2StepConfiguration] | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        app_id: str | None = ...,
        date_updated: int | None = ...,
        callback_id: str | None = ...,
        name: str | None = ...,
        updated_by: str | None = ...,
        step_configuration: list[dict[str, Any] | WorkflowV2StepConfiguration]
        | None = ...,
        **kwargs,
    ) -> None: ...

class AccountTypeRole:
    id: str | None
    name: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self, *, id: str | None = ..., name: str | None = ..., **kwargs
    ) -> None: ...

class SlackList:
    id: str | None
    unknown_fields: dict[str, Any]
    def __init__(self, *, id: str | None = ..., **kwargs) -> None: ...

class Entity:
    type: str | None
    user: User | None
    workspace: Location | None
    enterprise: Location | None
    channel: Channel | None
    file: File | None
    app: App | None
    message: Message | None
    huddle: Huddle | None
    role: Role | None
    usergroup: Usergroup | None
    workflow: Workflow | None
    barrier: InformationBarrier | None
    workflow_v2: WorkflowV2 | None
    account_type_role: AccountTypeRole | None
    list: SlackList | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        type: str | None = ...,
        user: User | dict[str, Any] | None = ...,
        workspace: Location | dict[str, Any] | None = ...,
        enterprise: Location | dict[str, Any] | None = ...,
        channel: Channel | dict[str, Any] | None = ...,
        file: File | dict[str, Any] | None = ...,
        app: App | dict[str, Any] | None = ...,
        message: Message | dict[str, Any] | None = ...,
        huddle: Huddle | dict[str, Any] | None = ...,
        role: Role | dict[str, Any] | None = ...,
        usergroup: Usergroup | dict[str, Any] | None = ...,
        workflow: Workflow | dict[str, Any] | None = ...,
        barrier: InformationBarrier | dict[str, Any] | None = ...,
        workflow_v2: WorkflowV2 | dict[str, Any] | None = ...,
        account_type_role: AccountTypeRole | dict[str, Any] | None = ...,
        list: SlackList | dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...

class Entry:
    id: str | None
    date_create: int | None
    action: str | None
    actor: Actor | None
    entity: Entity | None
    context: Context | None
    details: Details | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        id: str | None = ...,
        date_create: int | None = ...,
        action: str | None = ...,
        actor: Actor | dict[str, Any] | None = ...,
        entity: Entity | dict[str, Any] | None = ...,
        context: Context | dict[str, Any] | None = ...,
        details: Details | dict[str, Any] | None = ...,
        **kwargs,
    ) -> None: ...

class ResponseMetadata:
    next_cursor: str | None
    unknown_fields: dict[str, Any]
    def __init__(self, *, next_cursor: str | None = ..., **kwargs) -> None: ...

class LogsResponse:
    entries: list[Entry] | None
    response_metadata: ResponseMetadata | None
    ok: bool | None
    error: str | None
    needed: str | None
    provided: str | None
    unknown_fields: dict[str, Any]
    def __init__(
        self,
        *,
        entries: list[Entry | dict[str, Any]] | None = ...,
        response_metadata: ResponseMetadata | dict[str, Any] | None = ...,
        ok: bool | None = ...,
        error: str | None = ...,
        needed: str | None = ...,
        provided: str | None = ...,
        **kwargs,
    ) -> None: ...
