# bub-shrink

Automatic handoff on context limit for Bub.

## Description

`bub-shrink` reads the latest recorded token usage for the current session before each model run.

- If `BUB_SHRINK_CONTEXT_LIMIT` is unset, it forwards the original prompt unchanged.
- If the latest recorded usage is missing or does not exceed the configured limit, it forwards the original prompt unchanged.
- If the latest recorded usage exceeds the configured limit, it creates a `tape.handoff` anchor and then runs the original prompt with a context view that starts from the latest anchor.

The handoff metadata is fixed inside the plugin. This package does not implement recovery mode, retry-on-error behavior, or live token estimation for the current prompt.

## Configuration

Environment variables:

- `BUB_SHRINK_CONTEXT_LIMIT`: token threshold that enables automatic handoff based on the session's latest recorded token usage.

Example:

```bash
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
