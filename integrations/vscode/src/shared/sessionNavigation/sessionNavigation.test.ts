import { strict as assert } from "node:assert";
import test from "node:test";

import type { SagentSessionList } from "../sagentProtocol";
import {
  createInitialSessionNavigationState,
  sessionNavigationCreationState,
  sessionNavigationErrorState,
  sessionNavigationLoadingState,
  sessionNavigationSelectionState,
  sessionNavigationStateFromList,
} from "./sessionNavigation";
import { formatSessionAge } from "./sessionNavigationFormatting";

const listing: SagentSessionList = {
  activePath: "/sessions/current/session.jsonl",
  sessions: [
    {
      id: "current",
      path: "/sessions/current/session.jsonl",
      mtime: 1_735_000_000,
      title: "Fix the parser",
      records: 12,
      model: "gpt-5",
    },
    {
      id: "previous",
      path: "/sessions/previous/session.jsonl",
      mtime: 1_734_000_000,
      title: "Review the API",
      records: 8,
      model: "claude-sonnet",
    },
  ],
};

test("initial navigation state starts in a loading state", () => {
  assert.deepEqual(createInitialSessionNavigationState(), {
    status: "loading",
    sessions: [],
  });
});

test("a native Sagent listing is retained without losing session metadata", () => {
  const state = sessionNavigationStateFromList(listing);

  assert.equal(state.status, "ready");
  assert.equal(state.activePath, listing.activePath);
  assert.deepEqual(state.sessions, listing.sessions);
  assert.equal(state.sessions[1]?.records, 8);
  assert.equal(state.sessions[1]?.model, "claude-sonnet");
});

test("refreshing preserves the last successful listing", () => {
  const ready = sessionNavigationStateFromList(listing);

  assert.deepEqual(sessionNavigationLoadingState(ready), {
    ...ready,
    status: "loading",
    errorMessage: undefined,
  });
});

test("errors preserve discoverable sessions for retry", () => {
  const ready = sessionNavigationStateFromList(listing);
  const failed = sessionNavigationErrorState(ready, "Sagent is unavailable");

  assert.equal(failed.status, "error");
  assert.equal(failed.errorMessage, "Sagent is unavailable");
  assert.deepEqual(failed.sessions, listing.sessions);
  assert.equal(failed.pendingAction, undefined);
});

test("selection and creation expose typed parent intents", () => {
  const ready = sessionNavigationStateFromList(listing);

  assert.deepEqual(sessionNavigationSelectionState(ready, listing.sessions[1]!.path), {
    ...ready,
    pendingAction: {
      type: "select",
      sessionPath: listing.sessions[1]!.path,
    },
    errorMessage: undefined,
  });
  assert.deepEqual(sessionNavigationCreationState(ready), {
    ...ready,
    pendingAction: { type: "create" },
    errorMessage: undefined,
  });
});

test("session age formatting is stable at each display boundary", () => {
  const now = 2_000_000_000_000;

  assert.equal(formatSessionAge((now - 59_000) / 1000, now), "59s ago");
  assert.equal(formatSessionAge((now - 60 * 60_000) / 1000, now), "1h ago");
  assert.equal(formatSessionAge((now - 24 * 60 * 60_000) / 1000, now), "1d ago");
});
