from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, cast

from bub import BubFramework, hookimpl
from bub.builtin.agent import Agent
from bub.types import State
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from republic import TapeContext, TapeEntry

DEFAULT_HANDOFF_NAME = "auto-context-shrink"
DEFAULT_HANDOFF_SUMMARY = "Context exceeded configured limit; continue from latest handoff."
TRUNCATION_SUFFIX = "\n...[truncated by bub-shrink]"
HANDOFF_RETRY_STATE_KEY = "_shrink_handoff_retry"


class ShrinkSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BUB_SHRINK_",
        env_file=".env",
        extra="ignore",
    )

    context_limit: int | None = Field(default=None, gt=0)
    tool_result_max_chars: int | None = Field(default=None, gt=0)


class ShrinkPlugin:
    def __init__(self, framework: BubFramework) -> None:
        self.framework = framework

    @hookimpl(tryfirst=True)
    def build_tape_context(self) -> TapeContext:
        return TapeContext(select=_select_messages)

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
            "shrink.run_model.settings session_id={} context_limit={} tool_result_max_chars={}",
            session_id,
            settings.context_limit,
            settings.tool_result_max_chars,
        )
        try:
            logger.info("shrink.run_model.forward_original session_id={}", session_id)
            return await runtime_agent.run(session_id=session_id, prompt=prompt, state=state)
        except Exception as error:
            if not _is_context_overflow_error(error):
                logger.info(
                    "shrink.run_model.skip_handoff session_id={} reason=non_overflow_error error={}",
                    session_id,
                    str(error),
                )
                raise
            if state.get(HANDOFF_RETRY_STATE_KEY):
                logger.warning(
                    "shrink.run_model.skip_handoff session_id={} reason=retry_already_attempted error={}",
                    session_id,
                    str(error),
                )
                raise
            logger.warning(
                "shrink.run_model.overflow session_id={} context_limit={} error={}",
                session_id,
                settings.context_limit,
                str(error),
            )
            state[HANDOFF_RETRY_STATE_KEY] = True
            try:
                return await _run_with_handoff_context(
                    runtime_agent,
                    prompt,
                    session_id=session_id,
                    state=state,
                    error=str(error),
                    context_limit=settings.context_limit,
                )
            finally:
                state.pop(HANDOFF_RETRY_STATE_KEY, None)


def _load_settings() -> ShrinkSettings:
    settings = ShrinkSettings()
    logger.debug(
        "shrink.settings.loaded context_limit={} tool_result_max_chars={}",
        settings.context_limit,
        settings.tool_result_max_chars,
    )
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


async def _run_with_handoff_context(
    runtime_agent: Agent,
    prompt: str | list[dict],
    *,
    session_id: str,
    state: State,
    error: str,
    context_limit: int | None,
) -> str:
    if not prompt:
        return "error: empty prompt"

    tape = runtime_agent.tapes.session_tape(session_id, _workspace_from_state(state))
    tape.context = replace(tape.context, state=state)
    merge_back = not session_id.startswith("temp/")
    logger.info(
        "shrink.handoff_context.start session_id={} tape={} merge_back={} prompt_kind={} context_anchor={}",
        session_id,
        tape.name,
        merge_back,
        _prompt_kind(prompt),
        getattr(tape.context, "anchor", None),
    )
    async with runtime_agent.tapes.fork_tape(tape.name, merge_back=merge_back):
        await runtime_agent.tapes.ensure_bootstrap_anchor(tape.name)
        logger.warning(
            "shrink.handoff_context.handoff session_id={} tape={} context_limit={} handoff_name={} error={}",
            session_id,
            tape.name,
            context_limit,
            DEFAULT_HANDOFF_NAME,
            error,
        )
        await runtime_agent.tapes.handoff(
            tape.name,
            name=DEFAULT_HANDOFF_NAME,
            state={
                "summary": DEFAULT_HANDOFF_SUMMARY,
                "error": error,
                "limit": context_limit,
            },
        )
        if isinstance(prompt, str) and prompt.strip().startswith(","):
            logger.info("shrink.handoff_context.command session_id={} tape={}", session_id, tape.name)
            return await runtime_agent._run_command(tape=tape, line=prompt.strip())
        logger.info("shrink.handoff_context.agent_loop session_id={} tape={}", session_id, tape.name)
        return await runtime_agent._agent_loop(tape=tape, prompt=prompt)


def _select_messages(entries: list[TapeEntry], context: TapeContext) -> list[dict[str, Any]]:
    settings = _load_settings()
    messages: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []

    for entry in entries:
        match entry.kind:
            case "anchor":
                _append_anchor_entry(messages, entry)
            case "message":
                _append_message_entry(messages, entry)
            case "tool_call":
                pending_calls = _append_tool_call_entry(messages, entry)
            case "tool_result":
                _append_tool_result_entry(
                    messages,
                    pending_calls,
                    entry,
                    tool_result_max_chars=settings.tool_result_max_chars,
                )
                pending_calls = []
    return messages


def _append_anchor_entry(messages: list[dict[str, Any]], entry: TapeEntry) -> None:
    payload = entry.payload
    content = f"[Anchor created: {payload.get('name')}]: {json.dumps(payload.get('state'), ensure_ascii=False)}"
    messages.append({"role": "assistant", "content": content})


def _append_message_entry(messages: list[dict[str, Any]], entry: TapeEntry) -> None:
    payload = entry.payload
    if isinstance(payload, dict):
        messages.append(dict(payload))


def _append_tool_call_entry(messages: list[dict[str, Any]], entry: TapeEntry) -> list[dict[str, Any]]:
    calls = _normalize_tool_calls(entry.payload.get("calls"))
    if calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    return calls


def _append_tool_result_entry(
    messages: list[dict[str, Any]],
    pending_calls: list[dict[str, Any]],
    entry: TapeEntry,
    *,
    tool_result_max_chars: int | None,
) -> None:
    results = entry.payload.get("results")
    if not isinstance(results, list):
        return
    for index, result in enumerate(results):
        messages.append(
            _build_tool_result_message(
                result,
                pending_calls,
                index,
                tool_result_max_chars=tool_result_max_chars,
            )
        )


def _build_tool_result_message(
    result: object,
    pending_calls: list[dict[str, Any]],
    index: int,
    *,
    tool_result_max_chars: int | None,
) -> dict[str, Any]:
    rendered = _render_tool_result(result)
    message: dict[str, Any] = {"role": "tool", "content": rendered}
    if index < len(pending_calls):
        call = pending_calls[index]
        call_id = call.get("id")
        if isinstance(call_id, str) and call_id:
            message["tool_call_id"] = call_id

        function = call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                message["name"] = name
    if tool_result_max_chars is not None:
        message["content"] = _truncate_tool_result_content(
            str(message["content"]),
            tool_result_max_chars,
            tool_name=str(message.get("name", "-")),
        )
    return message


def _normalize_tool_calls(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            calls.append(dict(item))
    return calls


def _render_tool_result(result: object) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except TypeError:
        return str(result)


def _truncate_tool_result_content(content: str, max_chars: int, *, tool_name: str) -> str:
    if len(content) <= max_chars:
        return content
    if max_chars <= len(TRUNCATION_SUFFIX):
        truncated = TRUNCATION_SUFFIX[:max_chars]
    else:
        keep = max_chars - len(TRUNCATION_SUFFIX)
        truncated = content[:keep] + TRUNCATION_SUFFIX
    logger.warning(
        "shrink.tool_result.truncated tool_name={} original_chars={} truncated_chars={} max_chars={}",
        tool_name,
        len(content),
        len(truncated),
        max_chars,
    )
    return truncated


def _is_context_overflow_error(error: Exception) -> bool:
    text = str(error).casefold()
    markers = (
        "exceeded model token limit",
        "maximum context length",
        "context window exceeded",
        "context_length_exceeded",
        "prompt is too long",
        "too many tokens",
    )
    return any(marker in text for marker in markers)
