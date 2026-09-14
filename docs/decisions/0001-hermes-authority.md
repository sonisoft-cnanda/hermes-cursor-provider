# ADR 0001: Hermes owns the agent loop

- Status: Accepted
- Date: 2026-09-14

## Decision

Hermes is the control plane. It owns conversation state, system and developer
instructions, context management, memory, skills, tools, approvals, retries,
subagents, and auditing. The Cursor provider receives one complete request and
returns one assistant response or one or more requests for Hermes tools.

Cursor sessions are never resumed. Hermes tool definitions are serialized into
the request, but only Hermes executes them. Cursor-native tool events are a
protocol violation in `hermes` mode.

Each `hermes` request uses a fresh temporary workspace and a dedicated Cursor
home. The mode maps to Cursor Ask mode with its sandbox enabled and never uses
`--force`. Explicit workspaces are not allowed.

## Cursor CLI limitation

Cursor CLI print mode is an agent interface, not a public raw-inference API.
Ask mode is read-only but still exposes Cursor-native tools. The provider
therefore combines sandboxing, an empty workspace, a dedicated configuration
home, prompt instructions, event monitoring, and fail-closed output validation.
This containment reduces risk but cannot prove that no internal Cursor action
occurred before a tool event was observed.

The supported claim is therefore:

> Hermes remains authoritative and rejects Cursor-native agency at the provider
> boundary.

The project does not claim that Cursor CLI itself is a tool-free model API.

## Consequences

- Replacing Cursor with another inference provider does not move agent state or
  policy out of Hermes.
- Request isolation is preferred over Cursor session reuse.
- Unknown models, malformed tool calls, model mismatches, and native-tool events
  fail closed.
- `ask`, `plan`, and `agent` remain diagnostic compatibility modes outside the
  Hermes-authoritative contract. `agent` remains explicitly unsafe.
- Workspace context, Cursor rules, MCP, native images, persistent processes,
  and genuine incremental HTTP streaming remain separate future decisions.
