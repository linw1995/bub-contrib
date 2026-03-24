from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

from republic import TapeContext, TapeEntry

from bub_shrink import plugin


class FakeTapes:
    def __init__(self) -> None:
        self.session_tape_calls: list[tuple[str, Path]] = []
        self.handoff_calls: list[dict[str, object]] = []

    def session_tape(self, session_id: str, workspace: Path) -> SimpleNamespace:
        self.session_tape_calls.append((session_id, workspace))
        return SimpleNamespace(name=f"tape:{session_id}", context=TapeContext(state={}))

    async def handoff(self, tape_name: str, *, name: str, state: dict[str, object] | None = None) -> list[object]:
        self.handoff_calls.append({"tape_name": tape_name, "name": name, "state": state})
        return []

    @contextlib.asynccontextmanager
    async def fork_tape(self, tape_name: str, merge_back: bool = True):
        yield

    async def ensure_bootstrap_anchor(self, tape_name: str) -> None:
        return None


class FakeAgent:
    def __init__(self, *, result: str = "ok", error: Exception | None = None) -> None:
        self.tapes = FakeTapes()
        self.result = result
        self.error = error
        self.run_calls: list[dict[str, object]] = []

    async def run(self, *, session_id: str, prompt: str | list[dict], state: dict[str, object]) -> str:
        self.run_calls.append({"session_id": session_id, "prompt": prompt, "state": state})
        if self.error is not None:
            raise self.error
        return self.result


def test_select_messages_truncates_tool_result_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("BUB_SHRINK_TOOL_RESULT_MAX_CHARS", "64")
    entries = [
        TapeEntry.tool_call(
            [{"id": "call-1", "type": "function", "function": {"name": "web.fetch", "arguments": "{}"}}],
            run_id="run-1",
        ),
        TapeEntry.tool_result(["x" * 100], run_id="run-1"),
    ]

    messages = plugin._select_messages(entries, TapeContext())

    assert messages[0]["tool_calls"][0]["id"] == "call-1"
    assert messages[1]["role"] == "tool"
    assert messages[1]["name"] == "web.fetch"
    assert messages[1]["content"].endswith(plugin.TRUNCATION_SUFFIX)
    assert len(messages[1]["content"]) == 64


def test_select_messages_keeps_tool_result_when_limit_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("BUB_SHRINK_TOOL_RESULT_MAX_CHARS", raising=False)
    entries = [
        TapeEntry.tool_call(
            [{"id": "call-1", "type": "function", "function": {"name": "web.fetch", "arguments": "{}"}}],
            run_id="run-1",
        ),
        TapeEntry.tool_result(["short result"], run_id="run-1"),
    ]

    messages = plugin._select_messages(entries, TapeContext())

    assert messages[1]["content"] == "short result"


def test_run_model_passthrough_for_non_overflow_errors(monkeypatch) -> None:
    monkeypatch.setenv("BUB_SHRINK_TOOL_RESULT_MAX_CHARS", "1024")
    agent = FakeAgent(error=RuntimeError("network timeout"))
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]

    try:
        asyncio.run(shrink.run_model("hello", session_id="session-1", state={"_runtime_agent": agent}))  # type: ignore[arg-type]
    except RuntimeError as error:
        assert str(error) == "network timeout"
    else:
        raise AssertionError("expected RuntimeError")

    assert agent.run_calls == [{"session_id": "session-1", "prompt": "hello", "state": {"_runtime_agent": agent}}]
    assert agent.tapes.handoff_calls == []


def test_run_model_retries_with_handoff_on_context_overflow(monkeypatch) -> None:
    monkeypatch.setenv("BUB_SHRINK_CONTEXT_LIMIT", "200000")
    monkeypatch.setenv("BUB_SHRINK_TOOL_RESULT_MAX_CHARS", "4096")
    agent = FakeAgent(error=RuntimeError("invalid_input: exceeded model token limit"))
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]
    captured: list[dict[str, object]] = []

    async def fake_run_with_handoff_context(
        runtime_agent: object,
        prompt: str | list[dict],
        *,
        session_id: str,
        state: dict[str, object],
        error: str,
        context_limit: int | None,
    ) -> str:
        captured.append(
            {
                "runtime_agent": runtime_agent,
                "prompt": prompt,
                "session_id": session_id,
                "state": dict(state),
                "error": error,
                "context_limit": context_limit,
            }
        )
        return "continued output"

    monkeypatch.setattr(plugin, "_run_with_handoff_context", fake_run_with_handoff_context)

    result = asyncio.run(
        shrink.run_model("original prompt", session_id="session-1", state={"_runtime_agent": agent})  # type: ignore[arg-type]
    )

    assert result == "continued output"
    assert captured == [
        {
            "runtime_agent": agent,
            "prompt": "original prompt",
            "session_id": "session-1",
            "state": {"_runtime_agent": agent, plugin.HANDOFF_RETRY_STATE_KEY: True},
            "error": "invalid_input: exceeded model token limit",
            "context_limit": 200000,
        }
    ]


def test_run_model_does_not_retry_twice(monkeypatch) -> None:
    monkeypatch.setenv("BUB_SHRINK_CONTEXT_LIMIT", "200000")
    agent = FakeAgent(error=RuntimeError("invalid_input: exceeded model token limit"))
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]

    state = {"_runtime_agent": agent, plugin.HANDOFF_RETRY_STATE_KEY: True}  # type: ignore[arg-type]
    try:
        asyncio.run(shrink.run_model("hello", session_id="session-1", state=state))
    except RuntimeError as error:
        assert str(error) == "invalid_input: exceeded model token limit"
    else:
        raise AssertionError("expected RuntimeError")


def test_run_model_uses_runtime_workspace_for_tape_lookup(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BUB_SHRINK_TOOL_RESULT_MAX_CHARS", "1024")
    agent = FakeAgent(result="plain output")
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]

    result = asyncio.run(
        shrink.run_model(
            "hello",
            session_id="session-1",
            state={"_runtime_agent": agent, "_runtime_workspace": str(tmp_path)},  # type: ignore[arg-type]
        )
    )

    assert result == "plain output"
    assert agent.tapes.session_tape_calls == []


def test_run_model_raises_when_runtime_agent_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("BUB_SHRINK_TOOL_RESULT_MAX_CHARS", raising=False)
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]

    try:
        asyncio.run(shrink.run_model("hello", session_id="session-1", state={}))
    except RuntimeError as error:
        assert str(error) == "bub-shrink could not find an active model backend"
    else:
        raise AssertionError("expected RuntimeError")


def test_run_with_handoff_context_creates_handoff_inside_fork() -> None:
    events: list[tuple[str, object]] = []

    class TrackingTapes(FakeTapes):
        def session_tape(self, session_id: str, workspace: Path) -> SimpleNamespace:
            self.session_tape_calls.append((session_id, workspace))
            return SimpleNamespace(name=f"tape:{session_id}", context=TapeContext(state={}))

        @contextlib.asynccontextmanager
        async def fork_tape(self, tape_name: str, merge_back: bool = True):
            events.append(("fork_enter", merge_back))
            yield
            events.append(("fork_exit", merge_back))

        async def ensure_bootstrap_anchor(self, tape_name: str) -> None:
            events.append(("ensure_bootstrap_anchor", tape_name))

        async def handoff(
            self,
            tape_name: str,
            *,
            name: str,
            state: dict[str, object] | None = None,
        ) -> list[object]:
            events.append(("handoff", tape_name))
            return await super().handoff(tape_name, name=name, state=state)

    class TrackingAgent:
        def __init__(self) -> None:
            self.tapes = TrackingTapes()

        async def _agent_loop(self, *, tape: SimpleNamespace, prompt: str | list[dict]) -> str:
            events.append(("agent_loop", prompt))
            assert tape.context.state == {"flag": "set"}
            return "loop output"

        async def _run_command(self, tape: SimpleNamespace, *, line: str) -> str:
            raise AssertionError("command path should not be used")

    agent = TrackingAgent()

    result = asyncio.run(
        plugin._run_with_handoff_context(
            agent,  # type: ignore[arg-type]
            "original prompt",
            session_id="session-1",
            state={"flag": "set"},
            error="invalid_input: exceeded model token limit",
            context_limit=1024,
        )
    )

    assert result == "loop output"
    assert agent.tapes.handoff_calls == [
        {
            "tape_name": "tape:session-1",
            "name": plugin.DEFAULT_HANDOFF_NAME,
            "state": {
                "summary": plugin.DEFAULT_HANDOFF_SUMMARY,
                "error": "invalid_input: exceeded model token limit",
                "limit": 1024,
            },
        }
    ]
    assert events == [
        ("fork_enter", True),
        ("ensure_bootstrap_anchor", "tape:session-1"),
        ("handoff", "tape:session-1"),
        ("agent_loop", "original prompt"),
        ("fork_exit", True),
    ]
