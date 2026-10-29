"""Tests for a borrowed ChatGPT login, which only ever asks its lender."""

from __future__ import annotations

from dataclasses import dataclass, field

import asyncio
import time

import pytest

from sagent.model.exceptions import AuthError
from sagent.providers.openai.lease import BorrowedLogin, LeasedToken


@dataclass(slots=True, kw_only=True)
class _Lender:
    """Lends numbered tokens, a new one for each request, and keeps the requests."""

    lifetime_sec: float = 3_600.0
    asked: list[str] = field(default_factory=list[str])
    wait: asyncio.Event | None = None

    async def __call__(self, *, rejected: str) -> LeasedToken:
        self.asked.append(rejected)
        if self.wait is not None:
            await self.wait.wait()
        return LeasedToken(
            access_token=f"t{len(self.asked)}",
            account_id="acct",
            expires_at=time.time() + self.lifetime_sec,
        )


@pytest.mark.asyncio
async def test_a_token_is_borrowed_once_and_kept_while_fresh() -> None:
    lender = _Lender()
    borrowed = BorrowedLogin(lender)
    assert await borrowed.token() == ("t1", "acct")
    assert await borrowed.token() == ("t1", "acct")
    assert lender.asked == [""]


@pytest.mark.asyncio
async def test_a_token_about_to_expire_is_borrowed_again() -> None:
    lender = _Lender(lifetime_sec=299.0)
    borrowed = BorrowedLogin(lender)
    assert await borrowed.token() == ("t1", "acct")
    assert await borrowed.token() == ("t2", "acct")
    # Aging out by the local clock names no token as refused.
    assert lender.asked == ["", ""]


@pytest.mark.asyncio
async def test_renew_names_the_refused_token_to_the_lender() -> None:
    lender = _Lender()
    borrowed = BorrowedLogin(lender)
    await borrowed.token()
    await borrowed.renew()
    assert await borrowed.token() == ("t2", "acct")
    assert lender.asked == ["", "t1"]


@pytest.mark.asyncio
async def test_a_refusal_already_answered_by_another_call_borrows_nothing() -> None:
    lender = _Lender()
    borrowed = BorrowedLogin(lender)
    await borrowed.token()
    lender.wait = asyncio.Event()
    first = asyncio.create_task(borrowed.renew())
    second = asyncio.create_task(borrowed.renew())
    await asyncio.sleep(0)
    lender.wait.set()
    await asyncio.gather(first, second)
    # Both refused ``t1``; the second found ``t2`` already in hand.
    assert lender.asked == ["", "t1"]
    assert await borrowed.token() == ("t2", "acct")


@pytest.mark.asyncio
async def test_a_borrowed_login_cannot_sign_in() -> None:
    with pytest.raises(AuthError, match="lent by another process"):
        await BorrowedLogin(_Lender()).sign_in(show=list[str]().append)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
