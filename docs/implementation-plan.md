# Standalone Cursor Provider Implementation Plan

**Goal:** Ship an installable Cursor model-provider plugin for Hermes without modifying Hermes core.

**Architecture:** A loopback-only OpenAI-compatible HTTP bridge translates Hermes Chat Completions requests into one `cursor-agent -p --output-format stream-json` subprocess per request. A declarative Hermes provider profile points the standard OpenAI transport at that bridge. A package CLI installs the provider shim and a generated local bridge token under a chosen `HERMES_HOME`.

**Tech stack:** Python 3.11+, standard library HTTP server, pytest, Hermes public `ProviderProfile` API.

## Boundaries

- No edits to `NousResearch/hermes-agent` core.
- No shell-based subprocess invocation.
- Bind exactly to `127.0.0.1` with no non-loopback override.
- Authenticate bridge requests with a generated local bearer token.
- Never place `CURSOR_API_KEY` in process argv or logs.
- Build the Cursor subprocess environment from a strict allowlist; never copy
  the complete Hermes environment.
- Behavioral settings live in CLI arguments or TOML; environment variables are credentials only.
- Default workspace is an isolated temporary directory.
- Preserve the original PR's OpenAI response/tool-call contract and Cursor stream-json semantics.

## Task 1: Protocol contract

Create tests for:
- message-to-prompt ordering and Hermes tool schema
- tool-call extraction
- Cursor success/error stream accumulation
- usage conversion without cumulative-round inflation
- Cursor argv generation for agent/ask/plan modes
- API keys passed only in subprocess environment

Run focused tests and require expected import failures before implementation.

## Task 2: HTTP bridge contract

Create tests for:
- bearer authentication
- `/health`, `/v1/models`, and `/v1/chat/completions`
- OpenAI-compatible non-streaming and SSE responses
- invalid JSON and oversized request rejection
- loopback-only bind validation
- bounded concurrency, output, timeout and client-disconnect cancellation

Run focused tests RED, then implement the smallest application/server layer.

## Task 3: Installer and provider profile

Create tests for:
- provider files installed under `$HERMES_HOME/plugins/model-providers/cursor/`
- API-key profile pointing to the loopback bridge
- atomic/idempotent `.env` token insertion while preserving existing secrets
- conflict refusal, forced-install backup and failure rollback
- restrictive `.env` permissions
- uninstall scoped to plugin-owned files

Run focused tests RED, then implement installer/profile/CLI.

## Task 4: Integration verification

- Install into an isolated temporary Hermes home.
- Start the bridge against a deterministic fake Cursor executable.
- Exercise the bridge through the OpenAI SDK.
- Load the provider through an unmodified Hermes checkout.
- Run a real authenticated `cursor-agent` smoke when available.
- Run full pytest, compile, package build, install-from-wheel, and secret scan.

## Task 5: Review and release

- Freeze hashes and a clean diff.
- Run independent architecture/security/code review.
- Convert each concrete blocker into a RED test before repair.
- Create/push the standalone GitHub repository only after explicit release verification.
- Reply on PR #50215 with the standalone repository and installation instructions.
