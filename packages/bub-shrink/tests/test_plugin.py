from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from bub_shrink import plugin


class FakeTapes:
    def __init__(self, *, last_token_usage: int | None) -> None:
        self.last_token_usage = last_token_usage
        self.session_tape_calls: list[tuple[str, Path]] = []
        self.info_calls: list[str] = []
        self.handoff_calls: list[dict[str, object]] = []

    def session_tape(self, session_id: str, workspace: Path) -> SimpleNamespace:
        self.session_tape_calls.append((session_id, workspace))
        return SimpleNamespace(name=f"tape:{session_id}")

    async def info(self, tape_name: str) -> SimpleNamespace:
        self.info_calls.append(tape_name)
        return SimpleNamespace(last_token_usage=self.last_token_usage)

    async def handoff(self, tape_name: str, *, name: str, state: dict[str, object] | None = None) -> list[object]:
        self.handoff_calls.append({"tape_name": tape_name, "name": name, "state": state})
        return []


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

    result = asyncio.run(
        shrink.run_model("original prompt", session_id="session-1", state={"_runtime_agent": agent})  # type: ignore[arg-type]
    )

    assert result == "continued output"
    assert agent.tapes.info_calls == ["tape:session-1"]
    assert agent.tapes.handoff_calls == [
        {
            "tape_name": "tape:session-1",
            "name": plugin.DEFAULT_HANDOFF_NAME,
            "state": {
                "summary": plugin.DEFAULT_HANDOFF_SUMMARY,
                "usage": 1200,
                "limit": 1000,
            },
        }
    ]
    assert agent.run_calls == [
        {
            "session_id": "session-1",
            "prompt": plugin.CONTINUE_PROMPT,
            "state": {"_runtime_agent": agent},
        }
    ]


def test_run_model_uses_fixed_handoff_metadata(monkeypatch) -> None:
    monkeypatch.setenv("BUB_SHRINK_CONTEXT_LIMIT", "512")
    agent = FakeAgent(last_token_usage=4096)
    shrink = plugin.ShrinkPlugin(framework=object())  # type: ignore[arg-type]

    asyncio.run(
        shrink.run_model("original prompt", session_id="session-9", state={"_runtime_agent": agent})  # type: ignore[arg-type]
    )

    assert agent.tapes.handoff_calls == [
        {
            "tape_name": "tape:session-9",
            "name": plugin.DEFAULT_HANDOFF_NAME,
            "state": {
                "summary": plugin.DEFAULT_HANDOFF_SUMMARY,
                "usage": 4096,
                "limit": 512,
            },
        }
    ]
    assert agent.run_calls[0]["prompt"] == plugin.CONTINUE_PROMPT


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
