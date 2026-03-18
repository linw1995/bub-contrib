import textwrap
from typing import cast

from bub import hookimpl
from bub.builtin.agent import Agent
from bub.types import State


@hookimpl
def system_prompt(prompt: str, state: State) -> str:
    agent = cast(Agent, state["_runtime_agent"])
    session_id = state.get("session_id", "default")
    session_dir = agent.settings.home / "sessions"
    prompt_file = session_dir / session_id / "AGENTS.md"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    current_prompt = prompt_file.read_text() if prompt_file.exists() else ""
    prompt = textwrap.dedent(f"""\
    <session_instruct>
    The following is the content of {prompt_file}.
    You can edit this file to change the system prompt for the current session.

    Current session: {session_id}
    -----------------------------
    {current_prompt}
    </session_instruct>
    """)
    return prompt
