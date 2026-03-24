from __future__ import annotations

from pathlib import Path
from typing import cast

from bub import BubFramework, hookimpl
from bub.builtin.agent import CONTINUE_PROMPT, Agent
from bub.types import State
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_HANDOFF_NAME = "auto-context-shrink"
DEFAULT_HANDOFF_SUMMARY = "Context exceeded configured limit; continue from latest handoff."


class ShrinkSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BUB_SHRINK_",
        env_file=".env",
        extra="ignore",
    )

    context_limit: int | None = Field(default=None, gt=0)


class ShrinkPlugin:
    def __init__(self, framework: BubFramework) -> None:
        self.framework = framework

    @hookimpl(tryfirst=True)
    async def run_model(self, prompt: str | list[dict], session_id: str, state: State) -> str:
        runtime_agent = _runtime_agent_from_state(state)
        if runtime_agent is None:
            raise RuntimeError("bub-shrink could not find an active model backend")

        settings = _load_settings()
        should_handoff, tape_name, usage = await _should_handoff(
            runtime_agent,
            session_id=session_id,
            state=state,
            context_limit=settings.context_limit,
        )
        if should_handoff:
            await runtime_agent.tapes.handoff(
                tape_name,
                name=DEFAULT_HANDOFF_NAME,
                state={
                    "summary": DEFAULT_HANDOFF_SUMMARY,
                    "usage": usage,
                    "limit": settings.context_limit,
                },
            )
            return await runtime_agent.run(session_id=session_id, prompt=CONTINUE_PROMPT, state=state)

        return await runtime_agent.run(session_id=session_id, prompt=prompt, state=state)


def _load_settings() -> ShrinkSettings:
    return ShrinkSettings()


def _runtime_agent_from_state(state: State) -> Agent | None:
    agent = state.get("_runtime_agent")
    if agent is None:
        return None
    return cast(Agent, agent)


def _workspace_from_state(state: State) -> Path:
    raw = state.get("_runtime_workspace")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


async def _should_handoff(
    runtime_agent: Agent,
    *,
    session_id: str,
    state: State,
    context_limit: int | None,
) -> tuple[bool, str, int | None]:
    tape = runtime_agent.tapes.session_tape(session_id, _workspace_from_state(state))
    if context_limit is None:
        return False, tape.name, None

    info = await runtime_agent.tapes.info(tape.name)
    usage = getattr(info, "last_token_usage", None)
    if not isinstance(usage, int):
        return False, tape.name, None
    return usage > context_limit, tape.name, usage
