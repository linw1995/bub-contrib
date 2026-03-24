from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

from republic import TapeContext

from bub_shrink import plugin


class FakeTapes:
    def __init__(self, *, last_token_usage: int | None) -> None:
        self.last_token_usage = last_token_usage
        self.session_tape_calls: list[tuple[str, Path]] = []
        self.info_calls: list[str] = []
        self.handoff_calls: list[dict[str, object]] = []

    def session_tape(self, session_id: str, workspace: Path) -> SimpleNamespace:
        self.session_tape_calls.append((session_id, workspace))
        return SimpleNamespace(name=f"tape:{session_id}", context=TapeContext(state={}))

    async def info(self, tape_name: str) -> SimpleNamespace:
        self.info_calls.append(tape_name)
        return SimpleNamespace(last_token_usage=self.last_token_usage)

    async def handoff(self, tape_name: str, *, name: str, state: dict[str, object] | None = None) -> list[object]:
        self.handoff_calls.append({"tape_name": tape_name, "name": name, "state": state})
        return []

    @contextlib.asynccontextmanager
    async def fork_tape(self, tape_name: str, merge_back: bool = True):
        yield

    async def ensure_bootstrap_anchor(self, tape_name: str) -> None:
        return None


class FakeAgent:
    def __init__(self, *, last_token_usage: int | None, result: str = "ok") -> None:
        self.tapes = FakeTapes(last_token_usage=last_token_usage)
        self.result = result
        self.run_calls: list[dict[str, object]] = []

    async def run(self, *, session_id: str, prompt: str | list[dict], state: dict[str, object]) -> str:
        self.run_calls.append({"session_id": session_id, "prompt": prompt, "state": state})
        return self.result


def test_run_model_passthrough_when_context_limit_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("BUB_SHRINK_CONTEXT_LIMIT", raising=False)
    agent = FakeAgent(last_token_usage=9_999, result="plain output")
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]

    result = asyncio.run(
        shrink.run_model("hello", session_id="session-1", state={"_runtime_agent": agent})  # type: ignore[arg-type]
    )

    assert result == "plain output"
    assert agent.tapes.info_calls == []
    assert agent.tapes.handoff_calls == []
    assert agent.run_calls == [{"session_id": "session-1", "prompt": "hello", "state": {"_runtime_agent": agent}}]


def test_run_model_passthrough_when_usage_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("BUB_SHRINK_CONTEXT_LIMIT", "1000")
    agent = FakeAgent(last_token_usage=None, result="plain output")
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]

    result = asyncio.run(
        shrink.run_model("hello", session_id="session-1", state={"_runtime_agent": agent})  # type: ignore[arg-type]
    )

    assert result == "plain output"
    assert agent.tapes.info_calls == ["tape:session-1"]
    assert agent.tapes.handoff_calls == []
    assert agent.run_calls == [{"session_id": "session-1", "prompt": "hello", "state": {"_runtime_agent": agent}}]


def test_run_model_passthrough_when_usage_does_not_exceed_limit(monkeypatch) -> None:
    monkeypatch.setenv("BUB_SHRINK_CONTEXT_LIMIT", "1000")
    agent = FakeAgent(last_token_usage=1000, result="plain output")
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]

    result = asyncio.run(
        shrink.run_model("hello", session_id="session-1", state={"_runtime_agent": agent})  # type: ignore[arg-type]
    )

    assert result == "plain output"
    assert agent.tapes.info_calls == ["tape:session-1"]
    assert agent.tapes.handoff_calls == []
    assert agent.run_calls == [{"session_id": "session-1", "prompt": "hello", "state": {"_runtime_agent": agent}}]


def test_run_model_handoffs_and_continues_when_usage_exceeds_limit(monkeypatch) -> None:
    monkeypatch.setenv("BUB_SHRINK_CONTEXT_LIMIT", "1000")
    agent = FakeAgent(last_token_usage=1200, result="continued output")
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]
    captured: list[dict[str, object]] = []

    async def fake_run_with_handoff_context(
        runtime_agent: object,
        prompt: str | list[dict],
        *,
        session_id: str,
        state: dict[str, object],
        usage: int | None,
        context_limit: int | None,
    ) -> str:
        captured.append(
            {
                "runtime_agent": runtime_agent,
                "prompt": prompt,
                "session_id": session_id,
                "state": state,
                "usage": usage,
                "context_limit": context_limit,
            }
        )
        return "continued output"

    monkeypatch.setattr(plugin, "_run_with_handoff_context", fake_run_with_handoff_context)

    result = asyncio.run(
        shrink.run_model("original prompt", session_id="session-1", state={"_runtime_agent": agent})  # type: ignore[arg-type]
    )

    assert result == "continued output"
    assert agent.tapes.info_calls == ["tape:session-1"]
    assert agent.tapes.handoff_calls == []
    assert captured == [
        {
            "runtime_agent": agent,
            "session_id": "session-1",
            "prompt": "original prompt",
            "state": {"_runtime_agent": agent},
            "usage": 1200,
            "context_limit": 1000,
        }
    ]
    assert agent.run_calls == []


def test_run_model_uses_fixed_handoff_metadata(monkeypatch) -> None:
    monkeypatch.setenv("BUB_SHRINK_CONTEXT_LIMIT", "512")
    agent = FakeAgent(last_token_usage=4096)
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]
    captured: list[dict[str, object]] = []

    async def fake_run_with_handoff_context(
        runtime_agent: object,
        prompt: str | list[dict],
        *,
        session_id: str,
        state: dict[str, object],
        usage: int | None,
        context_limit: int | None,
    ) -> str:
        captured.append(
            {
                "runtime_agent": runtime_agent,
                "prompt": prompt,
                "session_id": session_id,
                "state": state,
                "usage": usage,
                "context_limit": context_limit,
            }
        )
        return "continued output"

    monkeypatch.setattr(plugin, "_run_with_handoff_context", fake_run_with_handoff_context)

    asyncio.run(
        shrink.run_model("original prompt", session_id="session-9", state={"_runtime_agent": agent})  # type: ignore[arg-type]
    )

    assert agent.tapes.handoff_calls == []
    assert captured[0]["prompt"] == "original prompt"
    assert captured[0]["usage"] == 4096
    assert captured[0]["context_limit"] == 512


def test_run_model_uses_runtime_workspace_for_tape_lookup(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BUB_SHRINK_CONTEXT_LIMIT", "1000")
    agent = FakeAgent(last_token_usage=500, result="plain output")
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]

    result = asyncio.run(
        shrink.run_model(
            "hello",
            session_id="session-1",
            state={"_runtime_agent": agent, "_runtime_workspace": str(tmp_path)},  # type: ignore[arg-type]
        )
    )

    assert result == "plain output"
    assert agent.tapes.session_tape_calls == [("session-1", tmp_path.resolve())]


def test_run_model_raises_when_runtime_agent_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("BUB_SHRINK_CONTEXT_LIMIT", raising=False)
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
            self.tapes = TrackingTapes(last_token_usage=4096)

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
            usage=4096,
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
                "usage": 4096,
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
