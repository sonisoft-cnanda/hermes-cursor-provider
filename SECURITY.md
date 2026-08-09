# Security

## Supported scope

The bridge is intended to run as the same local user as Hermes and Cursor. It
binds only to `127.0.0.1` and requires a token with at least 256 bits of generated
entropy on every route, including health and model listing.

Linux, macOS and WSL are supported. Native Windows installation and serving
fail closed until both Cursor CLI availability and Job Object cleanup are
implemented and tested.

## Trust boundaries

- Anyone with `CURSOR_BRIDGE_API_KEY` can send prompts to the bridge and consume the configured Cursor account.
- The listener is fixed to `127.0.0.1`; the CLI has no LAN-bind override.
- `--mode agent --workspace PATH` gives Cursor's internal tools direct access to that workspace. Hermes approvals do not intercept those internal operations.
- Cursor requires `--trust` for non-interactive use. The bridge acknowledges the
  bridge-created or operator-selected workspace in all modes, but only explicit
  `agent` mode receives `--force`.
- Cursor authentication is managed by `cursor-agent login` or `CURSOR_API_KEY`; the plugin does not store Cursor session files.
- The generated bridge token is stored in the selected Hermes `.env` with mode `0600`.
- Install and uninstall reject symlinks in every package-managed provider-path
  component, including when `--force` is selected.

## Secret handling

- Prompts are sent through subprocess stdin.
- API keys are not placed in subprocess argv.
- The subprocess starts from a strict allowlist of locale, home, path,
  temporary-directory and certificate variables. It never copies the full
  Hermes environment. The bridge token and unrelated API, cloud, wallet, SSH,
  proxy and loader variables are not inherited.
- HTTP clients receive a generic Cursor failure. Direct runner exceptions redact
  the configured Cursor API key.
- Combined Cursor stdout/stderr is capped at 16 MiB per invocation.
- Timeout, output overflow, shutdown and client disconnect terminate the whole
  POSIX Cursor process group.
- Cursor-emitted tool calls are forwarded only when their function name was
  supplied by Hermes in that request.
- HTTP authentication and bounded request admission happen before POST bodies
  are read. Per-connection timeouts and a fixed handler-thread cap bound slow
  local clients; saturation receives HTTP 429.
- Tests use generated or inert credentials only.

## Reporting

Do not open a public issue containing tokens, Cursor session data, private prompts, filesystem contents, or unredacted logs. Report the minimal reproduction and sanitized evidence to the repository maintainer through a private channel first.
