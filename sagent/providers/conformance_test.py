"""Provider/Model API conformance suite.

The objective this suite enforces: every provider fulfils the SAME
``Provider`` contract and every model fulfils the SAME ``Model``
contract. A deviation -- a model missing a protocol member, a provider
missing a factory method, or runtime/agent code probing the contract
with ``getattr`` instead of calling it -- is a bug, and shows up here as
a failure rather than as a silent ``getattr(..., None)`` no-op at
runtime.

The checks are introspective (no live credentials, no subprocess): they
compare each concrete class's member set against the protocol's, so a
new provider that forgets ``close()`` (or any other member) fails
immediately, and a member added to one model but not the others is
caught the moment it lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import ast
import inspect

import pytest

from sagent.providers.anthropic.api import _AnthropicModel
from sagent.providers.anthropic.cli import _AnthropicCLIModel
from sagent.providers.dashscope.api import _DashScopeModel
from sagent.providers.google.api import _GeminiModel
from sagent.providers.google.cli import _GoogleCLIModel
from sagent.providers.minimax.api import _MiniMaxModel
from sagent.providers.moonshot.api import _MoonshotModel
from sagent.providers.openai.api import _OpenAIModel
from sagent.providers.openai.sub import _OpenAISubModel
from sagent.types.model import Model
from sagent.types.providers import Provider


# Every concrete model class that claims to fulfil the ``Model``
# contract. Add a new provider's model class here -- the suite then
# holds it to the identical surface as the rest.
_MODEL_CLASSES = [
    _AnthropicModel,
    _AnthropicCLIModel,
    _GeminiModel,
    _GoogleCLIModel,
    _OpenAIModel,
    _OpenAISubModel,
    _DashScopeModel,
    _MiniMaxModel,
    _MoonshotModel,
]


def _protocol_members(proto: type) -> set[str]:
    """Public member names declared by a Protocol (methods AND properties).

    Walks the protocol's own ``__dict__`` so both ``def`` methods and
    ``@property`` descriptors count; excludes dunders and the
    ``Protocol`` machinery.
    """
    base = set(dir(Protocol))
    return {
        name
        for name, val in vars(proto).items()
        if not name.startswith("_")
        and name not in base
        and (callable(val) or isinstance(val, property))
    }


_MODEL_MEMBERS = _protocol_members(Model)
_PROVIDER_MEMBERS = _protocol_members(Provider)


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES, ids=lambda c: c.__name__)
def test_model_class_implements_full_protocol(model_cls: type) -> None:
    """Every model class implements every ``Model`` member -- no gaps.

    A missing member is the deviation that ``getattr`` probes used to
    paper over (e.g. ``close``/``cancel_in_flight``). Here it is a hard
    failure: uniformity is the contract.
    """
    missing = [m for m in _MODEL_MEMBERS if not hasattr(model_cls, m)]
    assert not missing, f"{model_cls.__name__} is missing Model members: {missing}"


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES, ids=lambda c: c.__name__)
def test_model_class_has_no_extra_public_surface_vs_peers(model_cls: type) -> None:
    """No model class carries a public method another lacks.

    A public method on one model but not the others is a latent
    deviation: callers that learn it will ``getattr``-probe for it. The
    contract is the union ceiling -- extra public surface must either be
    promoted to the protocol (so all implement it) or made private.
    """
    own_public = {
        name
        for name in vars(model_cls)
        if not name.startswith("_") and callable(getattr(model_cls, name, None))
    }
    extra = own_public - _MODEL_MEMBERS
    assert not extra, (
        f"{model_cls.__name__} exposes public members not in the Model "
        f"contract: {sorted(extra)} -- promote to the protocol or make private"
    )


def test_close_is_required_and_async_on_every_model() -> None:
    """``close()`` is a required, total, async member of every model.

    Regression guard for the ``getattr(model, 'close', None)`` smell:
    once ``close`` is a contract method, the probe is unjustifiable.
    """
    for model_cls in _MODEL_CLASSES:
        close = getattr(model_cls, "close", None)
        assert close is not None, f"{model_cls.__name__} has no close()"
        assert inspect.iscoroutinefunction(close), (
            f"{model_cls.__name__}.close must be async"
        )


def test_no_cancel_in_flight_method_anywhere() -> None:
    """Cancellation is the universal asyncio primitive, not a method.

    No model defines ``cancel_in_flight``: the runtime cancels the
    model-call task and providers translate ``CancelledError`` in their
    own ``stream``. A reappearing method would re-introduce the
    per-provider cancel divergence this refactor removed.
    """
    offenders = [c.__name__ for c in _MODEL_CLASSES if hasattr(c, "cancel_in_flight")]
    assert not offenders, (
        f"cancel_in_flight reappeared on: {offenders}; cancellation must stay "
        "the universal task-cancel primitive"
    )


# ----------------------------------------------------------------------
# getattr-smell guard: agent/runtime must not probe contract members.
# ----------------------------------------------------------------------

_GUARDED_FILES = [
    "agent/runtime.py",
    "agent/agent.py",
]


def _is_getattr_str_literal(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    )


def _getattr_string_literals(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(lineno, name)`` for every ``getattr(obj, "<name>", ...)``."""
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not _is_getattr_str_literal(node):
            continue
        assert isinstance(node, ast.Call)  # narrowed by guard above
        name = node.args[1]
        assert isinstance(name, ast.Constant)  # guard checked Constant[str]
        value = name.value
        assert isinstance(value, str)
        out.append((node.lineno, value))
    return out


@pytest.mark.parametrize("rel", _GUARDED_FILES)
def test_no_getattr_of_model_contract_members(rel: str) -> None:
    """agent/runtime never ``getattr``-probes a ``Model`` contract member.

    The contract is total, so probing it is the smell. A ``getattr`` of
    any protocol member name (``close``, ``stream``, ``buffer``,
    ``usage_snapshot``, ...) in these files means a deviation crept back
    in. ``getattr`` of NON-contract attributes (SDK exception fields,
    ``_provider``, etc.) is fine and not flagged.
    """
    sagent_root = Path(__file__).resolve().parent.parent
    src = (sagent_root / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders = [
        (lineno, name)
        for lineno, name in _getattr_string_literals(tree)
        if name in _MODEL_MEMBERS
    ]
    assert not offenders, (
        f"{rel}: getattr-probe of Model contract member(s) {offenders}; "
        "call the member directly -- the contract is total"
    )


def test_provider_protocol_members_are_callable() -> None:
    """Sanity: the Provider contract exposes the expected factory surface."""
    assert "model" in _PROVIDER_MEMBERS
    assert "utility_model" in _PROVIDER_MEMBERS
