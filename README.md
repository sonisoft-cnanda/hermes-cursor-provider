# Hermes Cursor Provider

Use models available through Cursor CLI as an inference backend for
[Hermes Agent](https://github.com/NousResearch/hermes-agent), while Hermes
remains responsible for orchestration, tools, memory, skills, approvals, and
conversation state.

It keeps Cursor-specific process handling outside Hermes core. Hermes talks to
a local OpenAI-compatible bridge; the bridge runs one stateless `cursor-agent`
request and translates Cursor's `stream-json` events back into chat completions
and requests for Hermes tools.

This repository is the out-of-tree continuation of [NousResearch/hermes-agent#50215](https://github.com/NousResearch/hermes-agent/pull/50215), following Hermes's third-party provider policy.

## Architecture

```text
Hermes Agent (control plane)
  -> history, memory, skills, MCP, tools and approvals
  -> OpenAI Chat Completions
  -> http://127.0.0.1:8765/v1
  -> authenticated local bridge
  -> isolated cursor-agent request
  -> selected Cursor model (inference plane)
```

The supported `hermes` mode intentionally does not implement:

```text
Hermes -> autonomous Cursor Agent -> shell/files/browser/MCP
```

Cursor CLI print mode is still an agent interface, not a public raw-model API.
The bridge contains it with Ask mode, Cursor sandboxing, a fresh empty workspace,
a dedicated Cursor home, explicit instructions, native-tool event rejection,
and strict output validation. See
[ADR 0001](docs/decisions/0001-hermes-authority.md) for the exact limitation.

The installed Hermes plugin is declarative and self-contained:

```text
$HERMES_HOME/plugins/model-providers/cursor/
├── __init__.py
└── plugin.yaml
```

It does not patch `agent_runtime_helpers.py`, `run_agent.py`, provider registries, or any other Hermes core file.

## Requirements

- Python 3.11 or newer
- A Hermes Agent release with user model-provider plugin discovery
- `cursor-agent` installed and authenticated

Release `0.1.0` is exercised against `cursor-agent 2026.07.23-e383d2b`.
Cursor's stream-json format is not a stable public protocol, so other CLI
versions may require compatibility fixes.

Supported platforms: Linux, macOS and WSL. Native Windows fails closed because
the Cursor CLI and descendant-process cleanup path are not yet verified there.

Check Cursor before installing:

```bash
cursor-agent --version
cursor-agent status
cursor-agent --list-models
```

If needed, authenticate with:

```bash
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
mkdir -p "$HERMES_HOME/cursor-home"
chmod 700 "$HERMES_HOME/cursor-home"
HOME="$HERMES_HOME/cursor-home" cursor-agent login
```

Using a dedicated home prevents normal global Cursor rules and MCP configuration
from entering the provider context. The bridge also supports `CURSOR_API_KEY`
as a credential in the selected Hermes `.env`.

## Install

From a checkout:

```bash
uv tool install .
hermes-cursor-provider install
hermes-cursor-provider doctor
hermes-cursor-provider doctor --probe-model cursor-grok-4.6-high
```

The optional probe performs a real completion through the same sandboxed path
used by `hermes` mode. It can consume Cursor usage; without it, doctor checks
the binary, authentication, model catalog, and standalone sandbox command only.

`install` performs two scoped operations:

1. Writes the self-contained provider profile under `$HERMES_HOME/plugins/model-providers/cursor/`.
2. Generates `CURSOR_BRIDGE_API_KEY` in `$HERMES_HOME/.env`, preserving existing entries and setting mode `0600`.

The generated bridge token is never printed.

Installation refuses to overwrite a conflicting existing `cursor` provider.
Inspect the existing files first; use `install --force` only when deliberate
replacement is intended. Forced replacement first moves the complete existing
provider directory to a sibling `cursor.backup-*` directory.
Installation and uninstallation reject symbolic links anywhere in the
package-managed provider path; `--force` does not bypass that boundary.

For a non-default Hermes home:

```bash
hermes-cursor-provider install --hermes-home /path/to/hermes-home
hermes-cursor-provider doctor --hermes-home /path/to/hermes-home
```

## Run the bridge

```bash
hermes-cursor-provider serve
```

The command reads `CURSOR_BRIDGE_API_KEY` and `CURSOR_API_KEY` from the process environment first, then from the selected Hermes `.env` without executing shell syntax.

Default endpoint:

```text
http://127.0.0.1:8765/v1
```

Default mode is `hermes`. Every request receives a new temporary workspace, so
Cursor does not see the calling repository or filesystem state from another
completion. It maps internally to Cursor Ask mode with `--sandbox enabled` and
never adds `--force`, resume, or continue.

Current Cursor releases require `--trust` for every non-interactive workspace.
The bridge acknowledges trust only for its bridge-created workspace. This
bypasses Cursor's interactive workspace prompt; it does not add `--force`.

Explicit writable Cursor-agent mode:

```bash
hermes-cursor-provider serve --mode agent
```

`agent` mode adds Cursor's `--force` flag and can execute commands or modify files inside the selected workspace. Treat it as an explicit unsafe operator choice. `ask` and `plan` never add `--force`.
The CLI prints a warning that Hermes approvals do not gate Cursor's internal
operations before starting in `agent` mode.

Explicit workspace access:

```bash
hermes-cursor-provider serve --mode agent --workspace /path/to/project
```

Giving a workspace to Cursor in `agent`, `ask`, or `plan` mode places that mode
outside the supported Hermes-authoritative contract. `hermes` mode rejects
`--workspace`.
Configured workspaces must already exist and the selected path itself must not
be a symbolic link. The bridge resolves the path before launching Cursor.

## Configure Hermes

```bash
hermes config set model.provider cursor
hermes config set model.default cursor-grok-4.6-high
hermes config set model.base_url http://127.0.0.1:8765/v1
```

Or select `cursor` through `hermes model` after installing the plugin.

Use `/v1/models` or `cursor-agent --list-models` to select an ID available to
the authenticated account. For example, the tested account exposes
`cursor-grok-4.6-high`, `cursor-grok-4.6-high-fast`,
`cursor-grok-4.6-medium`, and `cursor-grok-4.6-xhigh`.

Run a smoke test while the bridge is running:

```bash
hermes --oneshot 'Reply exactly CURSOR_OK' --provider cursor \
  --model cursor-grok-4.6-high --safe-mode
```

Configuration changes apply to new Hermes sessions. A running gateway must be restarted separately by its operator before it sees provider/config changes.

## CLI

```text
hermes-cursor-provider install [--hermes-home PATH] [--force]
hermes-cursor-provider serve [OPTIONS]
hermes-cursor-provider doctor [--hermes-home PATH]
hermes-cursor-provider uninstall [--hermes-home PATH] [--force]
```

Important `serve` options:

```text
--mode hermes|agent|ask|plan
--workspace PATH
--cursor-home PATH
--cursor-command COMMAND
--cursor-arg ARG
--timeout-seconds SECONDS
--max-concurrency COUNT
--port PORT
```

`--cursor-arg` is for wrapper-script arguments. Values that could override
bridge-owned authentication, mode, force, trust, model, workspace, prompt,
output-format, resume or header controls are rejected.

The bridge always binds exactly to `127.0.0.1`. There is no LAN-bind override.

## API surface

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

All routes require:

```http
Authorization: Bearer <CURSOR_BRIDGE_API_KEY>
```

Both normal responses and OpenAI-compatible SSE responses are supported. SSE is
currently emitted after the Cursor subprocess completes; `/health` reports
`streaming: false` so clients are not told this is real incremental streaming.

## Security model

- Fixed loopback-only binding with no host override
- Random 256-bit local bearer token required on every route
- No shell invocation
- Prompt passed through stdin, never argv
- `CURSOR_API_KEY` passed through subprocess environment, never argv
- Strict Cursor subprocess environment allowlist; unrelated Hermes, cloud,
  wallet, SSH and bridge credentials are not inherited
- Temporary isolated workspace by default
- Fresh workspace for every `hermes` request and dedicated Cursor home
- Cursor sandbox enabled and native Cursor tool events rejected in `hermes` mode
- Authentication and bounded admission occur before POST request bodies are read
- Bounded request body, handler threads, concurrency, subprocess output and wall timeout
- Slow HTTP headers and bodies are terminated by a per-connection timeout
- Whole process-group cleanup on timeout, output overflow, shutdown or client disconnect
- Generic HTTP error responses; direct runner errors redact the upstream key
- Tool calls are accepted only for function names offered in that request

See [SECURITY.md](SECURITY.md) for threat boundaries and reporting.

## Current limitations

- One `cursor-agent` process is created per completion request.
- Cursor built-in tool progress is not forwarded live into Hermes UI.
- Streaming is OpenAI-compatible but synthesized after completion.
- Image message parts are represented as placeholders; native image forwarding is not implemented.
- Cursor CLI does not expose a raw inference-only mode. The bridge fails closed
  on observed native-tool events, but cannot prove that Cursor performed no
  internal action before emitting an event.
- Prompt-token usage is estimated from the Hermes request. Cursor output usage is retained where it maps safely to OpenAI fields.
- The package does not install a background service. Supervise `serve` using your platform's normal user-service mechanism if needed.
- Browser clients are unsupported; no CORS policy is enabled.

## Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

The suite includes:

- Cursor stream-json parsing
- Prompt and Hermes tool-call translation
- Subprocess argv/environment boundaries
- Output caps, disconnect cancellation and descendant-process cleanup
- Bearer authentication and loopback enforcement
- Pre-body authentication, slow-client timeouts and total HTTP handler bounds
- Symlink-escape rejection for install and uninstall
- Tool-name allowlisting and error-secret redaction
- Real socket and OpenAI SDK compatibility
- Full fake-Cursor subprocess round trip
- Isolated provider installation and import

## Uninstall

```bash
hermes-cursor-provider uninstall
```

This removes only byte-matching plugin-owned files. It preserves `.env`,
credentials, unrelated plugins and locally modified provider files. Use
`uninstall --force` only after inspecting modified owned files. `pip uninstall`
alone does not remove the external provider shim. Remove the retained
`CURSOR_BRIDGE_API_KEY` line manually if it is no longer needed.

## Independence and trademarks

This is an independent community integration. It is not affiliated with or endorsed by Nous Research, Anysphere, Cursor, or their respective maintainers. Cursor and Hermes names belong to their respective owners.

## License

MIT
