import type { ReactElement } from "react";

import type {
  ModelSettingsAction,
  ModelSettingsPanelContract,
  ModelSettingsScope,
  ModelSettingName,
} from "../../../../src/shared/modelSettings/modelSettingsContract";
import {
  createModelSettingsViewModel,
  type ModelSettingsControl,
} from "../../../../src/shared/modelSettings/modelSettingsModel";
import { Button } from "../ui/Button";
import { Icon } from "../ui/Icon";
import { ModelSelectField } from "./ModelFields";

/**
 * Presentation-only model settings panel. Sagent supplies every option and
 * current value; actions are intents for the extension host to route through
 * a future Sagent-owned settings operation.
 */
export function ModelSettingsPanel({
  snapshot,
  onAction,
  onClose,
}: ModelSettingsPanelContract & { readonly onClose?: () => void }): ReactElement {
  const view = createModelSettingsViewModel(snapshot);

  if (view.status === "loading") {
    return <PanelMessage message="Loading Sagent model settings…" />;
  }

  if (view.status === "error") {
    return (
      <section
        aria-label="Sagent model settings error"
        className="flex flex-col gap-2 p-3 text-sm"
        role="alert"
      >
        <span className="text-[var(--vscode-errorForeground)]">
          {view.errorMessage ?? "Sagent model settings could not be loaded."}
        </span>
        <Button
          className="self-start"
          onClick={() => onAction({ type: "refresh" })}
        >
          <Icon name="refresh" />
          Retry
        </Button>
      </section>
    );
  }

  const renderControl = (control: ModelSettingsControl): ReactElement => (
    <SettingsControl
      key={control.name}
      control={control}
      onChange={(value) =>
        onAction({
          type: "set-setting",
          scope: view.selectedScope,
          name: control.name,
          value: settingValue(control.name, value),
        })
      }
    />
  );

  return (
    <section
      aria-label="Sagent model capabilities"
      className="flex min-w-0 flex-col gap-3 border-b border-sagent-border p-3"
    >
      <header className="flex items-start justify-end gap-2">
        <div className="flex items-center gap-1">
          <Button
            aria-label="Refresh Sagent model settings"
            className="!h-6 !min-h-6 !w-6 !p-0 translate-y-1"
            onClick={() => onAction({ type: "refresh" })}
            title="Refresh model settings"
            variant="ghost"
          >
            <Icon name="refresh" />
          </Button>
          {onClose && (
            <Button
              aria-label="Close model settings"
              className="!h-6 !min-h-6 !w-6 !p-0 translate-y-1"
              onClick={onClose}
              title="Close model settings"
              variant="ghost"
            >
              <Icon name="close" />
            </Button>
          )}
        </div>
      </header>

      {view.availableScopes.length > 1 && (
        <ModelSelectField
          label="Apply to"
          description="Choose whether changes affect the live agent or future sessions."
          value={view.selectedScope}
          options={view.availableScopes.map((scope) => ({
            value: scope,
            label: scope === "current-agent" ? "Current agent" : "New-session default",
          }))}
          onChange={(value) =>
            onAction({
              type: "select-scope",
              scope: value as ModelSettingsScope,
            })
          }
        />
      )}

      <div className="flex flex-col gap-2">
        <ModelSelectField
          label="Provider"
          description="Sagent's provider catalog for this agent."
          value={snapshot.provider.currentProvider}
          options={view.providerOptions}
          onChange={(provider) =>
            onAction({ type: "select-provider", scope: view.selectedScope, provider })
          }
        />
        <ModelSelectField
          label="Model"
          description="Exact model ID offered by the selected provider."
          value={snapshot.model.currentModel}
          options={view.modelOptions}
          onChange={(model) =>
            onAction({ type: "select-model", scope: view.selectedScope, model })
          }
        />
      </div>

      <div className="flex flex-col gap-2">
        {view.controls
          .filter((control) => !isAdvancedControl(control.name))
          .map(renderControl)}
        {view.controls.some((control) => isAdvancedControl(control.name)) && (
          <details className="mt-1">
            <summary className="cursor-pointer text-xs font-semibold text-sagent-muted-foreground">
              Advanced
            </summary>
            <div className="mt-2 flex flex-col gap-2">
              {view.controls
                .filter((control) => isAdvancedControl(control.name))
                .map(renderControl)}
            </div>
          </details>
        )}
      </div>

    </section>
  );
}

function isAdvancedControl(name: ModelSettingName): boolean {
  return name === "thinking.budget" || name === "thinking.output" || name === "service_tier";
}

function SettingsControl({
  control,
  onChange,
}: {
  readonly control: ModelSettingsControl;
  readonly onChange: (value: string) => void;
}): ReactElement {
  return (
    <ModelSelectField
      label={control.label}
      description={control.description}
      value={control.value}
      options={control.options}
      onChange={onChange}
    />
  );
}

function PanelMessage({ message }: { readonly message: string }): ReactElement {
  return (
    <section
      aria-label="Sagent model settings"
      className="p-3 text-sm text-sagent-muted-foreground"
      role="status"
    >
      {message}
    </section>
  );
}

function settingValue(
  name: ModelSettingName,
  value: string,
): string | number | boolean {
  if (name === "cache_ttl_sec") {
    return Number(value);
  }
  if (name === "manage_context_server_side") {
    return value === "true";
  }
  return value;
}
