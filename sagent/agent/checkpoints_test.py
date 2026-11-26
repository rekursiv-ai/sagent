"""Reports are independent per worker and user actions take priority."""

from sagent.agent.agent import Agent
from sagent.agent.checkpoints import Checkpoint, pending_text, report
from sagent.journal.types import UncategorizedRecord
from sagent.model.scripted import ScriptedProvider


def test_pending_prioritizes_actions_and_ignores_unrelated_or_malformed_records() -> (
    None
):
    session = Agent(model=ScriptedProvider()["scripted"]).session
    session.stand("other", UncategorizedRecord(kind="other"))
    session.stand("old", UncategorizedRecord(kind="work_checkpoint"))
    waiting = Checkpoint(progress="Fetched", pending="Review", user_action="")
    report(session, "a", waiting)
    report(session, "a", waiting)
    report(
        session,
        "z",
        Checkpoint(progress="Ready", pending="", user_action="Approve"),
    )
    assert pending_text(session, brief=True) == "z: Action needed: Approve"
    assert pending_text(session).splitlines() == [
        "z: Action needed: Approve",
        "  Ready",
        "a: Waiting: Review",
        "  Fetched",
    ]
    report(session, "z", None)
    assert pending_text(session, brief=True) == "a: Waiting: Review"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
