from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from bub import BubFramework, hookimpl
from bub.builtin.agent import Agent
from bub.builtin.context import _select_messages as _default_select_messages
from bub.types import State
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from republic import TapeContext, TapeEntry

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
        logger.info(
            "shrink.run_model.start session_id={} prompt_kind={} workspace={}",
            session_id,
            _prompt_kind(prompt),
            _workspace_from_state(state),
        )
        runtime_agent = _runtime_agent_from_state(state)
        if runtime_agent is None:
            logger.error("shrink.run_model.missing_runtime_agent session_id={}", session_id)
            raise RuntimeError("bub-shrink could not find an active model backend")

        settings = _load_settings()
        logger.info(
            "shrink.run_model.settings session_id={} context_limit={}",
            session_id,
            settings.context_limit,
        )
        should_handoff, tape_name, usage = await _should_handoff(
            runtime_agent,
            session_id=session_id,
            state=state,
            context_limit=settings.context_limit,
        )
        logger.info(
            "shrink.run_model.decision session_id={} tape={} usage={} context_limit={} should_handoff={}",
            session_id,
            tape_name,
            usage,
            settings.context_limit,
            should_handoff,
        )
        if should_handoff:
            logger.warning(
                "shrink.run_model.handoff session_id={} tape={} usage={} context_limit={} handoff_name={}",
                session_id,
                tape_name,
                usage,
                settings.context_limit,
                DEFAULT_HANDOFF_NAME,
            )
            await runtime_agent.tapes.handoff(
                tape_name,
                name=DEFAULT_HANDOFF_NAME,
                state={
                    "summary": DEFAULT_HANDOFF_SUMMARY,
                    "usage": usage,
                    "limit": settings.context_limit,
                },
            )
            logger.info(
                "shrink.run_model.run_with_handoff_context session_id={} tape={} prompt_kind={}",
                session_id,
                tape_name,
                _prompt_kind(prompt),
            )
            return await _run_with_handoff_context(
                runtime_agent,
                prompt,
                session_id=session_id,
                state=state,
            )

        logger.info("shrink.run_model.forward_original session_id={} tape={}", session_id, tape_name)
        return await runtime_agent.run(session_id=session_id, prompt=prompt, state=state)


def _load_settings() -> ShrinkSettings:
    settings = ShrinkSettings()
    logger.debug("shrink.settings.loaded context_limit={}", settings.context_limit)
    return settings


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


def _prompt_kind(prompt: str | list[dict]) -> str:
    if isinstance(prompt, str):
        return "text"
    return "parts"


def _select_messages_from_last_anchor(entries: list[TapeEntry], context: TapeContext) -> list[dict]:
    last_anchor_index = -1
    for index, entry in enumerate(entries):
        if entry.kind == "anchor":
            last_anchor_index = index
    if last_anchor_index < 0:
        logger.warning("shrink.context.no_anchor_found entries={}", len(entries))
        return _default_select_messages(entries, context)

    last_anchor = entries[last_anchor_index]
    logger.info(
        "shrink.context.after_last_anchor anchor_name={} entries_before={} entries_after={}",
        last_anchor.payload.get("name"),
        last_anchor_index,
        len(entries) - last_anchor_index,
    )
    return _default_select_messages(entries[last_anchor_index:], context)


async def _run_with_handoff_context(
    runtime_agent: Agent,
    prompt: str | list[dict],
    *,
    session_id: str,
    state: State,
) -> str:
    if not prompt:
        return "error: empty prompt"

    tape = runtime_agent.tapes.session_tape(session_id, _workspace_from_state(state))
    tape.context = replace(tape.context, select=_select_messages_from_last_anchor, state=state)
    merge_back = not session_id.startswith("temp/")
    logger.info(
        "shrink.handoff_context.start session_id={} tape={} merge_back={} prompt_kind={}",
        session_id,
        tape.name,
        merge_back,
        _prompt_kind(prompt),
    )
    async with runtime_agent.tapes.fork_tape(tape.name, merge_back=merge_back):
        await runtime_agent.tapes.ensure_bootstrap_anchor(tape.name)
        if isinstance(prompt, str) and prompt.strip().startswith(","):
            logger.info("shrink.handoff_context.command session_id={} tape={}", session_id, tape.name)
            return await runtime_agent._run_command(tape=tape, line=prompt.strip())
        logger.info("shrink.handoff_context.agent_loop session_id={} tape={}", session_id, tape.name)
        return await runtime_agent._agent_loop(tape=tape, prompt=prompt)


async def _should_handoff(
    runtime_agent: Agent,
    *,
    session_id: str,
    state: State,
    context_limit: int | None,
) -> tuple[bool, str, int | None]:
    tape = runtime_agent.tapes.session_tape(session_id, _workspace_from_state(state))
    logger.info("shrink.should_handoff.tape session_id={} tape={}", session_id, tape.name)
    if context_limit is None:
        logger.info("shrink.should_handoff.skip session_id={} tape={} reason=context_limit_unset", session_id, tape.name)
        return False, tape.name, None

    info = await runtime_agent.tapes.info(tape.name)
    usage = getattr(info, "last_token_usage", None)
    logger.info(
        "shrink.should_handoff.usage session_id={} tape={} last_token_usage={} context_limit={}",
        session_id,
        tape.name,
        usage,
        context_limit,
    )
    if not isinstance(usage, int):
        logger.info("shrink.should_handoff.skip session_id={} tape={} reason=usage_missing", session_id, tape.name)
        return False, tape.name, None
    if usage <= context_limit:
        logger.info(
            "shrink.should_handoff.skip session_id={} tape={} reason=usage_within_limit usage={} context_limit={}",
            session_id,
            tape.name,
            usage,
            context_limit,
        )
        return False, tape.name, usage
    logger.warning(
        "shrink.should_handoff.trigger session_id={} tape={} usage={} context_limit={}",
        session_id,
        tape.name,
        usage,
        context_limit,
    )
    return True, tape.name, usage
