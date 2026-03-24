# bub-shrink

Automatic handoff on context limit for Bub.

## Description

`bub-shrink` truncates oversized tool results before they are injected into model context.

- If `BUB_SHRINK_TOOL_RESULT_MAX_CHARS` is set, tool result messages longer than that value are truncated in prompt context.
- If the model call still fails with a context-limit style error, `bub-shrink` creates a `tape.handoff` anchor and retries once with the original prompt.

The handoff metadata is fixed inside the plugin. This package does not estimate the full request size up front; it focuses on trimming tool result payloads and only falls back to handoff after an actual overflow-style failure.

## Configuration

Environment variables:

- `BUB_SHRINK_TOOL_RESULT_MAX_CHARS`: maximum number of characters kept for each tool result in prompt context.
- `BUB_SHRINK_CONTEXT_LIMIT`: optional value recorded into handoff state for diagnostics when overflow fallback happens.

Example:

```bash
export BUB_SHRINK_TOOL_RESULT_MAX_CHARS=12000
export BUB_SHRINK_CONTEXT_LIMIT=120000
```

## Usage

Install via pip (from the monorepo):

```bash
uv pip install git+https://github.com/bubbuild/bub-contrib.git#subdirectory=packages/bub-shrink
```

Verify hook registration:

```bash
uv run bub hooks
```

You should see `shrink` registered for `run_model`.

## Development

- Requires Python 3.12+
- See the root README for workspace setup instructions

## Maintenance

Plugin contributors are encouraged to maintain this package and respond to issues or PRs. Code review standards are relaxed for contrib plugins, prioritizing practicality and safety.

## License

See [LICENSE](../../LICENSE).
