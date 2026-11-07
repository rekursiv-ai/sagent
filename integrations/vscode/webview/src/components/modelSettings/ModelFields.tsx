import type { ReactElement } from "react";

import type { ModelSettingsSelectOption } from "../../../../src/shared/modelSettings/modelSettingsModel";

export function ModelSelectField({
  label,
  description,
  value,
  options,
  onChange,
}: {
  readonly label: string;
  readonly description: string;
  readonly value: string;
  readonly options: readonly ModelSettingsSelectOption[];
  readonly onChange: (value: string) => void;
}): ReactElement {
  return (
    <label
      className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-baseline gap-x-2 gap-y-1 text-xs"
      title={description}
    >
      <span className="font-medium">{label}</span>
      <select
        aria-label={label}
        className="col-span-2 min-h-7 w-full rounded border border-sagent-border bg-sagent-input px-2 text-sagent-input-foreground focus-visible:outline-sagent-focus"
        disabled={options.length === 0}
        onChange={(event) => onChange(event.currentTarget.value)}
        value={value}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ReadOnlyModelField({
  label,
  description,
  value,
}: {
  readonly label: string;
  readonly description: string;
  readonly value: string;
}): ReactElement {
  return (
    <div
      className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-baseline gap-x-2 gap-y-1 text-xs"
      title={description}
    >
      <span className="font-medium">{label}</span>
      <output className="col-span-2 min-h-7 rounded border border-sagent-border bg-sagent-input px-2 py-1 text-sagent-input-foreground">
        {value}
      </output>
    </div>
  );
}
