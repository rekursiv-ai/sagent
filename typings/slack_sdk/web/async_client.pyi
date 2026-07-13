from collections.abc import Sequence
from io import IOBase
from typing import Any

import os

from slack_sdk.models.messages.chunk import Chunk
from slack_sdk.models.views import View
from slack_sdk.web.async_chat_stream import AsyncChatStream

from .async_base_client import AsyncBaseClient, AsyncSlackResponse
from ..models.attachments import Attachment
from ..models.blocks import Block, RichTextBlock
from ..models.metadata import EntityMetadata, EventAndEntityMetadata, Metadata

"""A Python module for interacting with Slack's Web API."""

class AsyncWebClient(AsyncBaseClient):
    async def admin_analytics_getFile(
        self,
        *,
        type: str,
        date: str | None = ...,
        metadata_only: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_approve(
        self,
        *,
        app_id: str | None = ...,
        request_id: str | None = ...,
        enterprise_id: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_approved_list(
        self,
        *,
        cursor: str | None = ...,
        limit: int | None = ...,
        enterprise_id: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_clearResolution(
        self,
        *,
        app_id: str,
        enterprise_id: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_requests_cancel(
        self,
        *,
        request_id: str,
        enterprise_id: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_requests_list(
        self,
        *,
        cursor: str | None = ...,
        limit: int | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_restrict(
        self,
        *,
        app_id: str | None = ...,
        request_id: str | None = ...,
        enterprise_id: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_restricted_list(
        self,
        *,
        cursor: str | None = ...,
        limit: int | None = ...,
        enterprise_id: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_uninstall(
        self,
        *,
        app_id: str,
        enterprise_id: str | None = ...,
        team_ids: str | Sequence[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_activities_list(
        self,
        *,
        app_id: str | None = ...,
        component_id: str | None = ...,
        component_type: str | None = ...,
        log_event_type: str | None = ...,
        max_date_created: int | None = ...,
        min_date_created: int | None = ...,
        min_log_level: str | None = ...,
        sort_direction: str | None = ...,
        source: str | None = ...,
        team_id: str | None = ...,
        trace_id: str | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_apps_config_lookup(
        self, *, app_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_apps_config_set(
        self,
        *,
        app_id: str,
        domain_restrictions: dict[str, Any] | None = ...,
        workflow_auth_strategy: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_auth_policy_getEntities(
        self,
        *,
        policy_name: str,
        cursor: str | None = ...,
        entity_type: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_auth_policy_assignEntities(
        self,
        *,
        entity_ids: str | Sequence[str],
        policy_name: str,
        entity_type: str,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_auth_policy_removeEntities(
        self,
        *,
        entity_ids: str | Sequence[str],
        policy_name: str,
        entity_type: str,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_createForObjects(
        self,
        *,
        object_id: str,
        salesforce_org_id: str,
        invite_object_team: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_linkObjects(
        self, *, channel: str, record_id: str, salesforce_org_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_unlinkObjects(
        self, *, channel: str, new_name: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_barriers_create(
        self,
        *,
        barriered_from_usergroup_ids: str | Sequence[str],
        primary_usergroup_id: str,
        restricted_subjects: str | Sequence[str],
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_barriers_delete(
        self, *, barrier_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_barriers_update(
        self,
        *,
        barrier_id: str,
        barriered_from_usergroup_ids: str | Sequence[str],
        primary_usergroup_id: str,
        restricted_subjects: str | Sequence[str],
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_barriers_list(
        self, *, cursor: str | None = ..., limit: int | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_create(
        self,
        *,
        is_private: bool,
        name: str,
        description: str | None = ...,
        org_wide: bool | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_delete(
        self, *, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_invite(
        self, *, channel_id: str, user_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_archive(
        self, *, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_unarchive(
        self, *, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_rename(
        self, *, channel_id: str, name: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_search(
        self,
        *,
        cursor: str | None = ...,
        limit: int | None = ...,
        query: str | None = ...,
        search_channel_types: str | Sequence[str] | None = ...,
        sort: str | None = ...,
        sort_dir: str | None = ...,
        team_ids: str | Sequence[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_convertToPrivate(
        self, *, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_convertToPublic(
        self, *, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_setConversationPrefs(
        self, *, channel_id: str, prefs: str | dict[str, str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_getConversationPrefs(
        self, *, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_disconnectShared(
        self,
        *,
        channel_id: str,
        leaving_team_ids: str | Sequence[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_lookup(
        self,
        *,
        last_message_activity_before: int,
        team_ids: str | Sequence[str],
        cursor: str | None = ...,
        limit: int | None = ...,
        max_member_count: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_ekm_listOriginalConnectedChannelInfo(
        self,
        *,
        channel_ids: str | Sequence[str] | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        team_ids: str | Sequence[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_restrictAccess_addGroup(
        self, *, channel_id: str, group_id: str, team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_restrictAccess_listGroups(
        self, *, channel_id: str, team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_restrictAccess_removeGroup(
        self, *, channel_id: str, group_id: str, team_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_setTeams(
        self,
        *,
        channel_id: str,
        org_channel: bool | None = ...,
        target_team_ids: str | Sequence[str] | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_getTeams(
        self,
        *,
        channel_id: str,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_getCustomRetention(
        self, *, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_removeCustomRetention(
        self, *, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_setCustomRetention(
        self, *, channel_id: str, duration_days: int, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_bulkArchive(
        self, *, channel_ids: Sequence[str] | str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_bulkDelete(
        self, *, channel_ids: Sequence[str] | str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_conversations_bulkMove(
        self, *, channel_ids: Sequence[str] | str, target_team_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_emoji_add(
        self, *, name: str, url: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_emoji_addAlias(
        self, *, alias_for: str, name: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_emoji_list(
        self, *, cursor: str | None = ..., limit: int | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_emoji_remove(
        self, *, name: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_emoji_rename(
        self, *, name: str, new_name: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_functions_list(
        self,
        *,
        app_ids: str | Sequence[str],
        team_id: str | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_functions_permissions_lookup(
        self, *, function_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_functions_permissions_set(
        self,
        *,
        function_id: str,
        visibility: str,
        user_ids: str | Sequence[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_roles_addAssignments(
        self,
        *,
        role_id: str,
        entity_ids: str | Sequence[str],
        user_ids: str | Sequence[str],
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_roles_listAssignments(
        self,
        *,
        role_ids: str | Sequence[str] | None = ...,
        entity_ids: str | Sequence[str] | None = ...,
        cursor: str | None = ...,
        limit: str | int | None = ...,
        sort_dir: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_roles_removeAssignments(
        self,
        *,
        role_id: str,
        entity_ids: str | Sequence[str],
        user_ids: str | Sequence[str],
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_users_session_reset(
        self,
        *,
        user_id: str,
        mobile_only: bool | None = ...,
        web_only: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_users_session_resetBulk(
        self,
        *,
        user_ids: str | Sequence[str],
        mobile_only: bool | None = ...,
        web_only: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_users_session_invalidate(
        self, *, session_id: str, team_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_users_session_list(
        self,
        *,
        cursor: str | None = ...,
        limit: int | None = ...,
        team_id: str | None = ...,
        user_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_teams_settings_setDefaultChannels(
        self, *, team_id: str, channel_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_users_session_getSettings(
        self, *, user_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_users_session_setSettings(
        self,
        *,
        user_ids: str | Sequence[str],
        desktop_app_browser_quit: bool | None = ...,
        duration: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_users_session_clearSettings(
        self, *, user_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_users_unsupportedVersions_export(
        self,
        *,
        date_end_of_support: str | int | None = ...,
        date_sessions_started: str | int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_inviteRequests_approve(
        self, *, invite_request_id: str, team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_inviteRequests_approved_list(
        self,
        *,
        cursor: str | None = ...,
        limit: int | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_inviteRequests_denied_list(
        self,
        *,
        cursor: str | None = ...,
        limit: int | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_inviteRequests_deny(
        self, *, invite_request_id: str, team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_inviteRequests_list(self, **kwargs) -> AsyncSlackResponse: ...
    async def admin_teams_admins_list(
        self,
        *,
        team_id: str,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_teams_create(
        self,
        *,
        team_domain: str,
        team_name: str,
        team_description: str | None = ...,
        team_discoverability: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_teams_list(
        self, *, cursor: str | None = ..., limit: int | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_teams_owners_list(
        self,
        *,
        team_id: str,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_teams_settings_info(
        self, *, team_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_teams_settings_setDescription(
        self, *, team_id: str, description: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_teams_settings_setDiscoverability(
        self, *, team_id: str, discoverability: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_teams_settings_setIcon(
        self, *, team_id: str, image_url: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_teams_settings_setName(
        self, *, team_id: str, name: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_usergroups_addChannels(
        self,
        *,
        channel_ids: str | Sequence[str],
        usergroup_id: str,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_usergroups_addTeams(
        self,
        *,
        usergroup_id: str,
        team_ids: str | Sequence[str],
        auto_provision: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_usergroups_listChannels(
        self,
        *,
        usergroup_id: str,
        include_num_members: bool | None = ...,
        team_id: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_usergroups_removeChannels(
        self, *, usergroup_id: str, channel_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_users_assign(
        self,
        *,
        team_id: str,
        user_id: str,
        channel_ids: str | Sequence[str] | None = ...,
        is_restricted: bool | None = ...,
        is_ultra_restricted: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_users_invite(
        self,
        *,
        team_id: str,
        email: str,
        channel_ids: str | Sequence[str],
        custom_message: str | None = ...,
        email_password_policy_enabled: bool | None = ...,
        guest_expiration_ts: str | float | None = ...,
        is_restricted: bool | None = ...,
        is_ultra_restricted: bool | None = ...,
        real_name: str | None = ...,
        resend: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_users_list(
        self,
        *,
        team_id: str | None = ...,
        include_deactivated_user_workspaces: bool | None = ...,
        is_active: bool | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_users_remove(
        self, *, team_id: str, user_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_users_setAdmin(
        self, *, team_id: str, user_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_users_setExpiration(
        self, *, expiration_ts: int, user_id: str, team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_users_setOwner(
        self, *, team_id: str, user_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_users_setRegular(
        self, *, team_id: str, user_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def admin_workflows_search(
        self,
        *,
        app_id: str | None = ...,
        collaborator_ids: str | Sequence[str] | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        no_collaborators: bool | None = ...,
        num_trigger_ids: int | None = ...,
        query: str | None = ...,
        sort: str | None = ...,
        sort_dir: str | None = ...,
        source: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_workflows_permissions_lookup(
        self,
        *,
        workflow_ids: str | Sequence[str],
        max_workflow_triggers: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_workflows_collaborators_add(
        self,
        *,
        collaborator_ids: str | Sequence[str],
        workflow_ids: str | Sequence[str],
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_workflows_collaborators_remove(
        self,
        *,
        collaborator_ids: str | Sequence[str],
        workflow_ids: str | Sequence[str],
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def admin_workflows_unpublish(
        self, *, workflow_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def api_test(
        self, *, error: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def apps_connections_open(
        self, *, app_token: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def apps_event_authorizations_list(
        self,
        *,
        event_context: str,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def apps_uninstall(
        self, *, client_id: str, client_secret: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def apps_manifest_create(
        self, *, manifest: str | dict[str, Any], **kwargs
    ) -> AsyncSlackResponse: ...
    async def apps_manifest_delete(
        self, *, app_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def apps_manifest_export(
        self, *, app_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def apps_manifest_update(
        self, *, app_id: str, manifest: str | dict[str, Any], **kwargs
    ) -> AsyncSlackResponse: ...
    async def apps_manifest_validate(
        self, *, manifest: str | dict[str, Any], app_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def apps_user_connection_update(
        self, *, user_id: str, status: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def tooling_tokens_rotate(
        self, *, refresh_token: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def assistant_threads_setStatus(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        status: str,
        loading_messages: list[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def assistant_threads_setTitle(
        self, *, channel_id: str, thread_ts: str, title: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def assistant_threads_setSuggestedPrompts(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        title: str | None = ...,
        prompts: list[dict[str, str]],
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def auth_revoke(
        self, *, test: bool | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def auth_test(self, **kwargs: Any) -> AsyncSlackResponse: ...
    async def auth_teams_list(
        self,
        cursor: str | None = ...,
        limit: int | None = ...,
        include_icon: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def bookmarks_add(
        self,
        *,
        channel_id: str,
        title: str,
        type: str,
        emoji: str | None = ...,
        entity_id: str | None = ...,
        link: str | None = ...,
        parent_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def bookmarks_edit(
        self,
        *,
        bookmark_id: str,
        channel_id: str,
        emoji: str | None = ...,
        link: str | None = ...,
        title: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def bookmarks_list(
        self, *, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def bookmarks_remove(
        self, *, bookmark_id: str, channel_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def bots_info(
        self, *, bot: str | None = ..., team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def calls_add(
        self,
        *,
        external_unique_id: str,
        join_url: str,
        created_by: str | None = ...,
        date_start: int | None = ...,
        desktop_app_join_url: str | None = ...,
        external_display_id: str | None = ...,
        title: str | None = ...,
        users: str | Sequence[dict[str, str]] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def calls_end(
        self, *, id: str, duration: int | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def calls_info(self, *, id: str, **kwargs) -> AsyncSlackResponse: ...
    async def calls_participants_add(
        self, *, id: str, users: str | Sequence[dict[str, str]], **kwargs
    ) -> AsyncSlackResponse: ...
    async def calls_participants_remove(
        self, *, id: str, users: str | Sequence[dict[str, str]], **kwargs
    ) -> AsyncSlackResponse: ...
    async def calls_update(
        self,
        *,
        id: str,
        desktop_app_join_url: str | None = ...,
        join_url: str | None = ...,
        title: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def canvases_create(
        self, *, title: str | None = ..., document_content: dict[str, str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def canvases_edit(
        self, *, canvas_id: str, changes: Sequence[dict[str, Any]], **kwargs
    ) -> AsyncSlackResponse: ...
    async def canvases_delete(
        self, *, canvas_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def canvases_access_set(
        self,
        *,
        canvas_id: str,
        access_level: str,
        channel_ids: Sequence[str] | str | None = ...,
        user_ids: Sequence[str] | str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def canvases_access_delete(
        self,
        *,
        canvas_id: str,
        channel_ids: Sequence[str] | str | None = ...,
        user_ids: Sequence[str] | str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def canvases_sections_lookup(
        self, *, canvas_id: str, criteria: dict[str, Any], **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_archive(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_create(self, *, name: str, **kwargs) -> AsyncSlackResponse: ...
    async def channels_history(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_info(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def channels_invite(
        self, *, channel: str, user: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_join(self, *, name: str, **kwargs) -> AsyncSlackResponse: ...
    async def channels_kick(
        self, *, channel: str, user: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_leave(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def channels_list(self, **kwargs) -> AsyncSlackResponse: ...
    async def channels_mark(
        self, *, channel: str, ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_rename(
        self, *, channel: str, name: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_replies(
        self, *, channel: str, thread_ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_setPurpose(
        self, *, channel: str, purpose: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_setTopic(
        self, *, channel: str, topic: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def channels_unarchive(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def chat_appendStream(
        self,
        *,
        channel: str,
        ts: str,
        markdown_text: str | None = ...,
        chunks: Sequence[dict | Chunk] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def chat_delete(
        self, *, channel: str, ts: str, as_user: bool | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def chat_deleteScheduledMessage(
        self,
        *,
        channel: str,
        scheduled_message_id: str,
        as_user: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def chat_getPermalink(
        self, *, channel: str, message_ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def chat_meMessage(
        self, *, channel: str, text: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def chat_postEphemeral(
        self,
        *,
        channel: str,
        user: str,
        text: str | None = ...,
        as_user: bool | None = ...,
        attachments: str | Sequence[dict | Attachment] | None = ...,
        blocks: str | Sequence[dict | Block] | None = ...,
        thread_ts: str | None = ...,
        icon_emoji: str | None = ...,
        icon_url: str | None = ...,
        link_names: bool | None = ...,
        username: str | None = ...,
        parse: str | None = ...,
        markdown_text: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def chat_postMessage(
        self,
        *,
        channel: str,
        text: str | None = ...,
        as_user: bool | None = ...,
        attachments: str | Sequence[dict[str, Any] | Attachment] | None = ...,
        blocks: str | Sequence[dict[str, Any] | Block] | None = ...,
        thread_ts: str | None = ...,
        reply_broadcast: bool | None = ...,
        unfurl_links: bool | None = ...,
        unfurl_media: bool | None = ...,
        container_id: str | None = ...,
        icon_emoji: str | None = ...,
        icon_url: str | None = ...,
        mrkdwn: bool | None = ...,
        link_names: bool | None = ...,
        username: str | None = ...,
        parse: str | None = ...,
        metadata: dict[str, Any] | Metadata | EventAndEntityMetadata | None = ...,
        markdown_text: str | None = ...,
        **kwargs: Any,
    ) -> AsyncSlackResponse: ...
    async def chat_scheduleMessage(
        self,
        *,
        channel: str,
        post_at: str | int,
        text: str | None = ...,
        as_user: bool | None = ...,
        attachments: str | Sequence[dict | Attachment] | None = ...,
        blocks: str | Sequence[dict | Block] | None = ...,
        thread_ts: str | None = ...,
        parse: str | None = ...,
        reply_broadcast: bool | None = ...,
        unfurl_links: bool | None = ...,
        unfurl_media: bool | None = ...,
        link_names: bool | None = ...,
        metadata: dict | Metadata | None = ...,
        markdown_text: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def chat_scheduledMessages_list(
        self,
        *,
        channel: str | None = ...,
        cursor: str | None = ...,
        latest: str | None = ...,
        limit: int | None = ...,
        oldest: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def chat_startStream(
        self,
        *,
        channel: str,
        thread_ts: str,
        markdown_text: str | None = ...,
        recipient_team_id: str | None = ...,
        recipient_user_id: str | None = ...,
        chunks: Sequence[dict | Chunk] | None = ...,
        task_display_mode: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def chat_stopStream(
        self,
        *,
        channel: str,
        ts: str,
        markdown_text: str | None = ...,
        blocks: str | Sequence[dict | Block] | None = ...,
        metadata: dict | Metadata | None = ...,
        chunks: Sequence[dict | Chunk] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def chat_stream(
        self,
        *,
        buffer_size: int = ...,
        channel: str,
        thread_ts: str,
        recipient_team_id: str | None = ...,
        recipient_user_id: str | None = ...,
        task_display_mode: str | None = ...,
        **kwargs,
    ) -> AsyncChatStream: ...
    async def chat_unfurl(
        self,
        *,
        channel: str | None = ...,
        ts: str | None = ...,
        source: str | None = ...,
        unfurl_id: str | None = ...,
        unfurls: dict[str, dict] | None = ...,
        metadata: dict | EventAndEntityMetadata | None = ...,
        user_auth_blocks: str | Sequence[dict | Block] | None = ...,
        user_auth_message: str | None = ...,
        user_auth_required: bool | None = ...,
        user_auth_url: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def chat_update(
        self,
        *,
        channel: str,
        ts: str,
        text: str | None = ...,
        attachments: str | Sequence[dict | Attachment] | None = ...,
        blocks: str | Sequence[dict | Block] | None = ...,
        as_user: bool | None = ...,
        file_ids: str | Sequence[str] | None = ...,
        link_names: bool | None = ...,
        parse: str | None = ...,
        reply_broadcast: bool | None = ...,
        metadata: dict | Metadata | None = ...,
        markdown_text: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_acceptSharedInvite(
        self,
        *,
        channel_name: str,
        channel_id: str | None = ...,
        invite_id: str | None = ...,
        free_trial_accepted: bool | None = ...,
        is_private: bool | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_approveSharedInvite(
        self, *, invite_id: str, target_team: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_archive(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_close(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_create(
        self,
        *,
        name: str,
        is_private: bool | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_declineSharedInvite(
        self, *, invite_id: str, target_team: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_externalInvitePermissions_set(
        self, *, action: str, channel: str, target_team: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_history(
        self,
        *,
        channel: str,
        cursor: str | None = ...,
        inclusive: bool | None = ...,
        include_all_metadata: bool | None = ...,
        latest: str | None = ...,
        limit: int | None = ...,
        oldest: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_info(
        self,
        *,
        channel: str,
        include_locale: bool | None = ...,
        include_num_members: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_invite(
        self,
        *,
        channel: str,
        users: str | Sequence[str],
        force: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_inviteShared(
        self,
        *,
        channel: str,
        emails: str | Sequence[str] | None = ...,
        user_ids: str | Sequence[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_join(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_kick(
        self, *, channel: str, user: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_leave(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_list(
        self,
        *,
        cursor: str | None = ...,
        exclude_archived: bool | None = ...,
        limit: int | None = ...,
        team_id: str | None = ...,
        types: str | Sequence[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_listConnectInvites(
        self,
        *,
        count: int | None = ...,
        cursor: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_mark(
        self, *, channel: str, ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_members(
        self,
        *,
        channel: str,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_open(
        self,
        *,
        channel: str | None = ...,
        return_im: bool | None = ...,
        users: str | Sequence[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_rename(
        self, *, channel: str, name: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_replies(
        self,
        *,
        channel: str,
        ts: str,
        cursor: str | None = ...,
        inclusive: bool | None = ...,
        include_all_metadata: bool | None = ...,
        latest: str | None = ...,
        limit: int | None = ...,
        oldest: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_requestSharedInvite_approve(
        self,
        *,
        invite_id: str,
        channel_id: str | None = ...,
        is_external_limited: str | None = ...,
        message: dict[str, Any] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_requestSharedInvite_deny(
        self, *, invite_id: str, message: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_requestSharedInvite_list(
        self,
        *,
        cursor: str | None = ...,
        include_approved: bool | None = ...,
        include_denied: bool | None = ...,
        include_expired: bool | None = ...,
        invite_ids: str | Sequence[str] | None = ...,
        limit: int | None = ...,
        user_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def conversations_setPurpose(
        self, *, channel: str, purpose: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_setTopic(
        self, *, channel: str, topic: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_unarchive(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def conversations_canvases_create(
        self, *, channel_id: str, document_content: dict[str, str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def dialog_open(
        self, *, dialog: dict[str, Any], trigger_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def dnd_endDnd(self, **kwargs) -> AsyncSlackResponse: ...
    async def dnd_endSnooze(self, **kwargs) -> AsyncSlackResponse: ...
    async def dnd_info(
        self, *, team_id: str | None = ..., user: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def dnd_setSnooze(
        self, *, num_minutes: int | str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def dnd_teamInfo(
        self, users: str | Sequence[str], team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def emoji_list(
        self, include_categories: bool | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def entity_presentDetails(
        self,
        trigger_id: str,
        metadata: dict | EntityMetadata | None = ...,
        user_auth_required: bool | None = ...,
        user_auth_url: str | None = ...,
        error: dict[str, Any] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_comments_delete(
        self, *, file: str, id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def files_delete(self, *, file: str, **kwargs) -> AsyncSlackResponse: ...
    async def files_info(
        self,
        *,
        file: str,
        count: int | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        page: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_list(
        self,
        *,
        channel: str | None = ...,
        count: int | None = ...,
        page: int | None = ...,
        show_files_hidden_by_limit: bool | None = ...,
        team_id: str | None = ...,
        ts_from: str | None = ...,
        ts_to: str | None = ...,
        types: str | Sequence[str] | None = ...,
        user: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_remote_info(
        self, *, external_id: str | None = ..., file: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def files_remote_list(
        self,
        *,
        channel: str | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        ts_from: str | None = ...,
        ts_to: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_remote_add(
        self,
        *,
        external_id: str,
        external_url: str,
        title: str,
        filetype: str | None = ...,
        indexable_file_contents: str | bytes | IOBase | None = ...,
        preview_image: str | bytes | IOBase | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_remote_update(
        self,
        *,
        external_id: str | None = ...,
        external_url: str | None = ...,
        file: str | None = ...,
        title: str | None = ...,
        filetype: str | None = ...,
        indexable_file_contents: str | None = ...,
        preview_image: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_remote_remove(
        self, *, external_id: str | None = ..., file: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def files_remote_share(
        self,
        *,
        channels: str | Sequence[str],
        external_id: str | None = ...,
        file: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_revokePublicURL(
        self, *, file: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def files_sharedPublicURL(
        self, *, file: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def files_upload(
        self,
        *,
        file: str | bytes | IOBase | None = ...,
        content: str | bytes | None = ...,
        filename: str | None = ...,
        filetype: str | None = ...,
        initial_comment: str | None = ...,
        thread_ts: str | None = ...,
        title: str | None = ...,
        channels: str | Sequence[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_upload_v2(
        self,
        *,
        filename: str | None = ...,
        file: str | bytes | IOBase | os.PathLike | None = ...,
        content: str | bytes | None = ...,
        title: str | None = ...,
        alt_txt: str | None = ...,
        snippet_type: str | None = ...,
        file_uploads: list[dict[str, Any]] | None = ...,
        channel: str | None = ...,
        channels: list[str] | None = ...,
        initial_comment: str | None = ...,
        thread_ts: str | None = ...,
        request_file_info: bool = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_getUploadURLExternal(
        self,
        *,
        filename: str,
        length: int,
        alt_txt: str | None = ...,
        snippet_type: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def files_completeUploadExternal(
        self,
        *,
        files: list[dict[str, str]],
        channel_id: str | None = ...,
        channels: list[str] | None = ...,
        initial_comment: str | None = ...,
        thread_ts: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def functions_completeSuccess(
        self, *, function_execution_id: str, outputs: dict[str, Any], **kwargs
    ) -> AsyncSlackResponse: ...
    async def functions_completeError(
        self, *, function_execution_id: str, error: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def groups_archive(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def groups_create(self, *, name: str, **kwargs) -> AsyncSlackResponse: ...
    async def groups_createChild(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def groups_history(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def groups_info(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def groups_invite(
        self, *, channel: str, user: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def groups_kick(
        self, *, channel: str, user: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def groups_leave(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def groups_list(self, **kwargs) -> AsyncSlackResponse: ...
    async def groups_mark(
        self, *, channel: str, ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def groups_open(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def groups_rename(
        self, *, channel: str, name: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def groups_replies(
        self, *, channel: str, thread_ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def groups_setPurpose(
        self, *, channel: str, purpose: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def groups_setTopic(
        self, *, channel: str, topic: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def groups_unarchive(
        self, *, channel: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def im_close(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def im_history(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def im_list(self, **kwargs) -> AsyncSlackResponse: ...
    async def im_mark(
        self, *, channel: str, ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def im_open(self, *, user: str, **kwargs) -> AsyncSlackResponse: ...
    async def im_replies(
        self, *, channel: str, thread_ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def migration_exchange(
        self,
        *,
        users: str | Sequence[str],
        team_id: str | None = ...,
        to_old: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def mpim_close(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def mpim_history(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def mpim_list(self, **kwargs) -> AsyncSlackResponse: ...
    async def mpim_mark(
        self, *, channel: str, ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def mpim_open(
        self, *, users: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def mpim_replies(
        self, *, channel: str, thread_ts: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def oauth_v2_access(
        self,
        *,
        client_id: str,
        client_secret: str,
        code: str | None = ...,
        redirect_uri: str | None = ...,
        grant_type: str | None = ...,
        refresh_token: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def oauth_access(
        self,
        *,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def oauth_v2_exchange(
        self, *, token: str, client_id: str, client_secret: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def openid_connect_token(
        self,
        client_id: str,
        client_secret: str,
        code: str | None = ...,
        redirect_uri: str | None = ...,
        grant_type: str | None = ...,
        refresh_token: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def openid_connect_userInfo(self, **kwargs) -> AsyncSlackResponse: ...
    async def pins_add(
        self, *, channel: str, timestamp: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def pins_list(self, *, channel: str, **kwargs) -> AsyncSlackResponse: ...
    async def pins_remove(
        self, *, channel: str, timestamp: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def reactions_add(
        self, *, channel: str, name: str, timestamp: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def reactions_get(
        self,
        *,
        channel: str | None = ...,
        file: str | None = ...,
        file_comment: str | None = ...,
        full: bool | None = ...,
        timestamp: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def reactions_list(
        self,
        *,
        count: int | None = ...,
        cursor: str | None = ...,
        full: bool | None = ...,
        limit: int | None = ...,
        page: int | None = ...,
        team_id: str | None = ...,
        user: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def reactions_remove(
        self,
        *,
        name: str,
        channel: str | None = ...,
        file: str | None = ...,
        file_comment: str | None = ...,
        timestamp: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def reminders_add(
        self,
        *,
        text: str,
        time: str,
        team_id: str | None = ...,
        user: str | None = ...,
        recurrence: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def reminders_complete(
        self, *, reminder: str, team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def reminders_delete(
        self, *, reminder: str, team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def reminders_info(
        self, *, reminder: str, team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def reminders_list(
        self, *, team_id: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def rtm_connect(
        self,
        *,
        batch_presence_aware: bool | None = ...,
        presence_sub: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def rtm_start(
        self,
        *,
        batch_presence_aware: bool | None = ...,
        include_locale: bool | None = ...,
        mpim_aware: bool | None = ...,
        no_latest: bool | None = ...,
        no_unreads: bool | None = ...,
        presence_sub: bool | None = ...,
        simple_latest: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def search_all(
        self,
        *,
        query: str,
        count: int | None = ...,
        highlight: bool | None = ...,
        page: int | None = ...,
        sort: str | None = ...,
        sort_dir: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def search_files(
        self,
        *,
        query: str,
        count: int | None = ...,
        highlight: bool | None = ...,
        page: int | None = ...,
        sort: str | None = ...,
        sort_dir: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def search_messages(
        self,
        *,
        query: str,
        count: int | None = ...,
        cursor: str | None = ...,
        highlight: bool | None = ...,
        page: int | None = ...,
        sort: str | None = ...,
        sort_dir: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def slackLists_access_delete(
        self,
        *,
        list_id: str,
        channel_ids: list[str] | None = ...,
        user_ids: list[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def slackLists_access_set(
        self,
        *,
        list_id: str,
        access_level: str,
        channel_ids: list[str] | None = ...,
        user_ids: list[str] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def slackLists_create(
        self,
        *,
        name: str,
        description_blocks: str | Sequence[dict | RichTextBlock] | None = ...,
        schema: list[dict[str, Any]] | None = ...,
        copy_from_list_id: str | None = ...,
        include_copied_list_records: bool | None = ...,
        todo_mode: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def slackLists_download_get(
        self, *, list_id: str, job_id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def slackLists_download_start(
        self, *, list_id: str, include_archived: bool | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def slackLists_items_create(
        self,
        *,
        list_id: str,
        duplicated_item_id: str | None = ...,
        parent_item_id: str | None = ...,
        initial_fields: list[dict[str, Any]] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def slackLists_items_delete(
        self, *, list_id: str, id: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def slackLists_items_deleteMultiple(
        self, *, list_id: str, ids: list[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def slackLists_items_info(
        self,
        *,
        list_id: str,
        id: str,
        include_is_subscribed: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def slackLists_items_list(
        self,
        *,
        list_id: str,
        limit: int | None = ...,
        cursor: str | None = ...,
        archived: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def slackLists_items_update(
        self, *, list_id: str, cells: list[dict[str, Any]], **kwargs
    ) -> AsyncSlackResponse: ...
    async def slackLists_update(
        self,
        *,
        id: str,
        name: str | None = ...,
        description_blocks: str | Sequence[dict | RichTextBlock] | None = ...,
        todo_mode: bool | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def stars_add(
        self,
        *,
        channel: str | None = ...,
        file: str | None = ...,
        file_comment: str | None = ...,
        timestamp: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def stars_list(
        self,
        *,
        count: int | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        page: int | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def stars_remove(
        self,
        *,
        channel: str | None = ...,
        file: str | None = ...,
        file_comment: str | None = ...,
        timestamp: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def team_accessLogs(
        self,
        *,
        before: int | str | None = ...,
        count: int | str | None = ...,
        page: int | str | None = ...,
        team_id: str | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def team_billableInfo(
        self, *, team_id: str | None = ..., user: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def team_billing_info(self, **kwargs) -> AsyncSlackResponse: ...
    async def team_externalTeams_disconnect(
        self, *, target_team: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def team_externalTeams_list(
        self,
        *,
        connection_status_filter: str | None = ...,
        slack_connect_pref_filter: Sequence[str] | None = ...,
        sort_direction: str | None = ...,
        sort_field: str | None = ...,
        workspace_filter: Sequence[str] | None = ...,
        cursor: str | None = ...,
        limit: int | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def team_info(
        self, *, team: str | None = ..., domain: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def team_integrationLogs(
        self,
        *,
        app_id: str | None = ...,
        change_type: str | None = ...,
        count: int | str | None = ...,
        page: int | str | None = ...,
        service_id: str | None = ...,
        team_id: str | None = ...,
        user: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def team_profile_get(
        self, *, visibility: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def team_preferences_list(self, **kwargs) -> AsyncSlackResponse: ...
    async def usergroups_create(
        self,
        *,
        name: str,
        channels: str | Sequence[str] | None = ...,
        description: str | None = ...,
        handle: str | None = ...,
        include_count: bool | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def usergroups_disable(
        self,
        *,
        usergroup: str,
        include_count: bool | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def usergroups_enable(
        self,
        *,
        usergroup: str,
        include_count: bool | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def usergroups_list(
        self,
        *,
        include_count: bool | None = ...,
        include_disabled: bool | None = ...,
        include_users: bool | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def usergroups_update(
        self,
        *,
        usergroup: str,
        channels: str | Sequence[str] | None = ...,
        description: str | None = ...,
        handle: str | None = ...,
        include_count: bool | None = ...,
        name: str | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def usergroups_users_list(
        self,
        *,
        usergroup: str,
        include_disabled: bool | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def usergroups_users_update(
        self,
        *,
        usergroup: str,
        users: str | Sequence[str],
        include_count: bool | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def users_conversations(
        self,
        *,
        cursor: str | None = ...,
        exclude_archived: bool | None = ...,
        limit: int | None = ...,
        team_id: str | None = ...,
        types: str | Sequence[str] | None = ...,
        user: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def users_deletePhoto(self, **kwargs) -> AsyncSlackResponse: ...
    async def users_getPresence(self, *, user: str, **kwargs) -> AsyncSlackResponse: ...
    async def users_identity(self, **kwargs) -> AsyncSlackResponse: ...
    async def users_info(
        self, *, user: str, include_locale: bool | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def users_list(
        self,
        *,
        cursor: str | None = ...,
        include_locale: bool | None = ...,
        limit: int | None = ...,
        team_id: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def users_lookupByEmail(
        self, *, email: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def users_setPhoto(
        self,
        *,
        image: str | IOBase,
        crop_w: int | str | None = ...,
        crop_x: int | str | None = ...,
        crop_y: int | str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def users_setPresence(
        self, *, presence: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def users_discoverableContacts_lookup(
        self, email: str, **kwargs
    ) -> AsyncSlackResponse: ...
    async def users_profile_get(
        self, *, user: str | None = ..., include_labels: bool | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def users_profile_set(
        self,
        *,
        name: str | None = ...,
        value: str | None = ...,
        user: str | None = ...,
        profile: dict | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def views_open(
        self,
        *,
        trigger_id: str | None = ...,
        interactivity_pointer: str | None = ...,
        view: dict | View,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def views_push(
        self,
        *,
        trigger_id: str | None = ...,
        interactivity_pointer: str | None = ...,
        view: dict | View,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def views_update(
        self,
        *,
        view: dict | View,
        external_id: str | None = ...,
        view_id: str | None = ...,
        hash: str | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
    async def views_publish(
        self, *, user_id: str, view: dict | View, hash: str | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def workflows_featured_add(
        self, *, channel_id: str, trigger_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def workflows_featured_list(
        self, *, channel_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def workflows_featured_remove(
        self, *, channel_id: str, trigger_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def workflows_featured_set(
        self, *, channel_id: str, trigger_ids: str | Sequence[str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def workflows_stepCompleted(
        self, *, workflow_step_execute_id: str, outputs: dict | None = ..., **kwargs
    ) -> AsyncSlackResponse: ...
    async def workflows_stepFailed(
        self, *, workflow_step_execute_id: str, error: dict[str, str], **kwargs
    ) -> AsyncSlackResponse: ...
    async def workflows_updateStep(
        self,
        *,
        workflow_step_edit_id: str,
        inputs: dict[str, Any] | None = ...,
        outputs: list[dict[str, str]] | None = ...,
        **kwargs,
    ) -> AsyncSlackResponse: ...
