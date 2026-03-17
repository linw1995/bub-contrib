# bub-contrib

Contrib packages for the `bub` ecosystem.

## Packages

| Package                             | Bub Plugin Entry Point | Description                                                                                                                               |
| ----------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `packages/bub-codex`                | `codex`                | Provides a `run_model` hook that delegates model execution to the Codex CLI.                                                              |
| `packages/bub-tg-feed`              | `tg-feed`              | Provides an AMQP-based channel adapter for Telegram feed messages.                                                                        |
| `packages/bub-schedule`             | `schedule`             | Provides scheduling channel/tools backed by APScheduler with a JSON job store.                                                            |
| `packages/bub-tapestore-sqlalchemy` | `tapestore-sqlalchemy` | Provides a SQLAlchemy-backed tape store for Bub conversation history.                                                                     |
| `packages/bub-tapestore-sqlite`     | `tapestore-sqlite`     | Provides a SQLite-backed tape store for Bub conversation history.                                                                         |
| `packages/bub-discord`              | `discord`              | Provides a Discord channel adapter for Bub message IO.                                                                                    |
| `packages/bub-dingtalk`             | `dingtalk`             | Provides a DingTalk Stream Mode channel adapter for Bub message IO.                                                                       |
| `packages/bub-web-search`           | `web-search`           | Provides a `web.search` tool backed by the Ollama web search API. Registers the tool only when `BUB_SEARCH_OLLAMA_API_KEY` is configured. |
| `packages/bub-feishu`               | `feishu`               | Provides a Feishu channel adapter for Bub message IO.                                                                                     |
| `packages/bub-session-prompt`       | `session-prompt`       | Provides a session-specific system prompt sourced from `~/.bub/sessions/<session_id>/AGENTS.md`.                                          |

## Prerequisites

- Python 3.12+ (workspace root)
- `uv` (recommended)

## Usage

To install an individual package, run:

```bash
uv pip install git+https://github.com/bubbuild/bub-contrib.git#subdirectory=packages/<package-name>
```

## Development Setup

Install all workspace dependencies:

```bash
uv sync
```

## License

This repository is licensed under [LICENSE](./LICENSE).
