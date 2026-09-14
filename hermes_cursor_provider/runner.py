"""Safe Cursor CLI subprocess runner."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from .protocol import (
    CursorProtocolAdapter,
    CursorProtocolError,
    estimate_prompt_tokens,
    format_messages_as_prompt,
)

PREFERRED_MODELS = (
    "auto",
    "cursor-grok-4.6-high",
    "cursor-grok-4.6-high-fast",
    "cursor-grok-4.6-medium",
    "cursor-grok-4.6-xhigh",
    "composer-2.5",
    "composer-2.5-fast",
    "composer-2",
    "composer-2-fast",
    "gpt-5.5-medium",
    "gpt-5.5-medium-fast",
    "gpt-5.5-high",
    "gpt-5.5-high-fast",
    "claude-opus-4-7-high",
    "gemini-3.1-pro",
)
FALLBACK_MODELS = PREFERRED_MODELS[:5]
_VALID_MODES = frozenset({"hermes", "agent", "ask", "plan"})
_SAFE_ENV_NAMES = frozenset(
    {
        "HOME",
        "LANG",
        "LOGNAME",
        "PATH",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
    }
)
_FORBIDDEN_EXTRA_FLAGS = frozenset(
    {
        "--api-key",
        "--approve-mcps",
        "--force",
        "--header",
        "--mode",
        "--model",
        "--output-format",
        "--resume",
        "--sandbox",
        "--trust",
        "--workspace",
        "--yolo",
        "-H",
        "-p",
    }
)


class CursorRunnerError(RuntimeError):
    """Base class for safe bridge error mapping."""


class CursorAuthenticationError(CursorRunnerError):
    pass


class CursorUnavailableError(CursorRunnerError):
    pass


class CursorModelUnavailableError(CursorRunnerError):
    pass


class CursorTimeoutError(CursorRunnerError):
    pass


class CursorOutputLimitError(CursorRunnerError):
    pass


class CursorCancelledError(CursorRunnerError):
    pass


@dataclass(frozen=True)
class RunnerConfig:
    command: str = "cursor-agent"
    extra_args: tuple[str, ...] = ()
    mode: str = "hermes"
    workspace: str | None = None
    cursor_home: str | None = None
    timeout_seconds: float = 1800.0
    max_output_bytes: int = 16 * 1024 * 1024
    cursor_api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("Cursor command cannot be empty")
        if self.mode not in _VALID_MODES:
            raise ValueError("Cursor mode must be one of: hermes, agent, ask, plan")
        if self.mode == "hermes" and self.workspace is not None:
            raise ValueError("hermes mode does not permit an explicit workspace")
        if self.timeout_seconds <= 0:
            raise ValueError("Cursor timeout_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("Cursor max_output_bytes must be positive")
        for value in self.extra_args:
            argument = str(value)
            if "\x00" in argument or any(
                argument == flag or argument.startswith(f"{flag}=")
                for flag in _FORBIDDEN_EXTRA_FLAGS
            ):
                raise ValueError(f"Unsafe Cursor extra argument is not allowed: {argument}")


def build_cursor_argv(
    *,
    command: str,
    extra_args: Iterable[str],
    model: str,
    workspace: str,
    mode: str,
) -> list[str]:
    """Build argv without secrets and without invoking a shell."""
    if mode not in _VALID_MODES:
        raise ValueError("Cursor mode must be one of: hermes, agent, ask, plan")
    argv = [
        command,
        *[str(value) for value in extra_args],
        "-p",
        "--output-format",
        "stream-json",
        "--model",
        model or "auto",
        "--workspace",
        workspace,
        "--trust",
    ]
    if mode == "agent":
        argv.append("--force")
    if mode == "hermes":
        argv.extend(["--mode", "ask", "--sandbox", "enabled"])
    elif mode in {"ask", "plan"}:
        argv.extend(["--mode", mode])
    return argv


def parse_model_catalog(output: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for raw in output.splitlines():
        line = raw.strip()
        if not line or " - " not in line:
            continue
        model_id = line.split(" - ", 1)[0].strip().split()[0]
        key = model_id.lower()
        if model_id and key not in seen:
            seen.add(key)
            ids.append(model_id)
    by_lower = {model.lower(): model for model in ids}
    preferred = [by_lower[model.lower()] for model in PREFERRED_MODELS if model.lower() in by_lower]
    preferred_keys = {model.lower() for model in preferred}
    return preferred + [model for model in ids if model.lower() not in preferred_keys]


def _redact(text: str, secrets: Iterable[str | None]) -> str:
    clean = text
    for secret in secrets:
        if secret:
            clean = clean.replace(secret, "[REDACTED]")
    return clean[:4000]


class CursorRunner:
    """Run one Cursor CLI subprocess per completion request."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        resolved_command = shutil.which(config.command)
        if not resolved_command:
            raise ValueError(
                f"Cursor CLI command '{config.command}' was not found; install cursor-agent first"
            )
        command_path = Path(resolved_command).expanduser().resolve(strict=True)
        if not command_path.is_file() or not os.access(command_path, os.X_OK):
            raise ValueError(f"Cursor CLI command is not executable: {command_path}")
        self._command = str(command_path)
        self._ephemeral_workspace: str | None = None
        self._auth_checked = False
        self._active: set[subprocess.Popen[Any]] = set()
        self._active_lock = threading.Lock()
        self._models_cache: tuple[float, list[str]] | None = None
        self._model_lock = threading.Lock()
        if config.cursor_home:
            cursor_home = Path(config.cursor_home).expanduser()
            cursor_home.mkdir(parents=True, exist_ok=True, mode=0o700)
            if cursor_home.is_symlink() or not cursor_home.is_dir():
                raise ValueError("Cursor home must be a real directory, not a symbolic link")
            os.chmod(cursor_home, 0o700)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        with self._active_lock:
            active = tuple(self._active)
        for process in active:
            self._terminate_process_tree(process)
        if self._ephemeral_workspace:
            shutil.rmtree(self._ephemeral_workspace, ignore_errors=True)
            self._ephemeral_workspace = None

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
        """Terminate the exact Cursor request process group, never a name match."""
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

    def _communicate_bounded(
        self,
        process: subprocess.Popen[Any],
        prompt: str,
        cancel_event: threading.Event | None,
        timeout_seconds: float | None = None,
    ) -> tuple[str, str, int]:
        """Drain both pipes with a combined byte cap and hard request deadline."""
        stdout_parts: list[bytes] = []
        stderr_parts: list[bytes] = []
        output_exceeded = threading.Event()
        native_tool_attempted = threading.Event()
        byte_lock = threading.Lock()
        total_bytes = 0

        def drain(stream: Any, parts: list[bytes], inspect_native_tools: bool = False) -> None:
            nonlocal total_bytes
            if stream is None:
                return
            line_buffer = bytearray()
            monitor = CursorProtocolAdapter(reject_native_tools=True)
            read_chunk = getattr(stream, "read1", stream.read)
            while True:
                chunk = read_chunk(65536)
                if not chunk:
                    return
                with byte_lock:
                    remaining = self.config.max_output_bytes - total_bytes
                    if remaining <= 0:
                        output_exceeded.set()
                        return
                    accepted = chunk[:remaining]
                    parts.append(accepted)
                    total_bytes += len(accepted)
                    if inspect_native_tools and accepted:
                        line_buffer.extend(accepted)
                        while b"\n" in line_buffer:
                            raw_line, _, remainder = line_buffer.partition(b"\n")
                            line_buffer = bytearray(remainder)
                            try:
                                event = json.loads(raw_line.decode("utf-8"))
                                if isinstance(event, dict):
                                    monitor.feed(event)
                            except CursorProtocolError:
                                native_tool_attempted.set()
                                return
                            except (UnicodeDecodeError, ValueError):
                                pass
                    if len(chunk) > remaining:
                        output_exceeded.set()
                        return

        def send_prompt() -> None:
            if process.stdin is None:
                return
            try:
                process.stdin.write(prompt.encode("utf-8"))
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        threads = [
            threading.Thread(
                target=drain,
                args=(
                    process.stdout,
                    stdout_parts,
                    self.config.mode == "hermes",
                ),
                daemon=True,
            ),
            threading.Thread(target=drain, args=(process.stderr, stderr_parts), daemon=True),
            threading.Thread(target=send_prompt, daemon=True),
        ]
        for thread in threads:
            thread.start()

        request_timeout = timeout_seconds or self.config.timeout_seconds
        deadline = time.monotonic() + request_timeout
        failure: CursorRunnerError | None = None
        while process.poll() is None:
            if native_tool_attempted.is_set():
                failure = CursorRunnerError("Cursor attempted native tool use in hermes mode")
                break
            if cancel_event is not None and cancel_event.is_set():
                failure = CursorCancelledError("Cursor request was cancelled")
                break
            if output_exceeded.is_set():
                failure = CursorOutputLimitError(
                    f"Cursor CLI exceeded the {self.config.max_output_bytes} byte output limit"
                )
                break
            if time.monotonic() >= deadline:
                failure = CursorTimeoutError(
                    f"Cursor CLI exceeded the {request_timeout:g}s request timeout"
                )
                break
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass

        if failure is not None:
            self._terminate_process_tree(process)
        for thread in threads:
            thread.join(timeout=2)
        if native_tool_attempted.is_set():
            raise CursorProtocolError("Cursor attempted native tool use in hermes mode")
        if output_exceeded.is_set() and failure is None:
            failure = CursorOutputLimitError(
                f"Cursor CLI exceeded the {self.config.max_output_bytes} byte output limit"
            )
        if failure is not None:
            raise failure
        stdout = b"".join(stdout_parts).decode("utf-8", errors="replace")
        stderr = b"".join(stderr_parts).decode("utf-8", errors="replace")
        return stdout, stderr, process.returncode if process.returncode is not None else -1

    def _workspace(self) -> str:
        if self.config.workspace:
            path = Path(self.config.workspace).expanduser()
            if path.is_symlink():
                raise ValueError("Cursor workspace must not be a symbolic link")
            try:
                resolved = path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError("Cursor workspace must be an existing directory") from exc
            if not resolved.is_dir():
                raise ValueError("Cursor workspace must be an existing directory")
            return str(resolved)
        if not self._ephemeral_workspace:
            self._ephemeral_workspace = tempfile.mkdtemp(prefix="hermes-cursor-provider-")
        return self._ephemeral_workspace

    @contextmanager
    def _request_workspace(self) -> Iterator[str]:
        if self.config.mode == "hermes":
            with tempfile.TemporaryDirectory(prefix="hermes-cursor-request-") as workspace:
                yield workspace
            return
        yield self._workspace()

    def _env(self) -> dict[str, str]:
        env = {
            name: value
            for name, value in os.environ.items()
            if name in _SAFE_ENV_NAMES or name.startswith("LC_")
        }
        if self.config.cursor_api_key:
            env["CURSOR_API_KEY"] = self.config.cursor_api_key
        if self.config.cursor_home:
            home = str(Path(self.config.cursor_home).expanduser().resolve(strict=True))
            env["HOME"] = home
            env["XDG_CONFIG_HOME"] = str(Path(home) / ".config")
            env["XDG_CACHE_HOME"] = str(Path(home) / ".cache")
            env["XDG_DATA_HOME"] = str(Path(home) / ".local" / "share")
        env["NO_COLOR"] = "1"
        env.setdefault("TERM", "dumb")
        return env

    def _run_control_command(
        self,
        arguments: list[str],
        *,
        timeout_seconds: float,
    ) -> tuple[str, str, int]:
        process: subprocess.Popen[Any] | None = None
        try:
            process = subprocess.Popen(
                [self._command, *arguments],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env(),
                start_new_session=os.name == "posix",
            )
            with self._active_lock:
                self._active.add(process)
            return self._communicate_bounded(
                process,
                "",
                None,
                timeout_seconds=timeout_seconds,
            )
        finally:
            if process is not None:
                self._terminate_process_tree(process)
                with self._active_lock:
                    self._active.discard(process)

    def _preflight_auth(self) -> None:
        if self.config.cursor_api_key or self._auth_checked:
            return
        try:
            stdout, stderr, returncode = self._run_control_command(
                [*self.config.extra_args, "status"],
                timeout_seconds=min(self.config.timeout_seconds, 30.0),
            )
            if returncode == 0:
                self._auth_checked = True
                return
        except (OSError, RuntimeError) as exc:
            stdout = ""
            stderr = str(exc)
        detail = _redact(
            " ".join(value for value in (stdout, stderr) if value),
            (self.config.cursor_api_key,),
        )
        raise CursorAuthenticationError(
            "Cursor CLI is not authenticated; run `cursor-agent login` or configure CURSOR_API_KEY. "
            + detail
        )

    def list_models(self, *, refresh: bool = False) -> list[str]:
        with self._model_lock:
            now = time.monotonic()
            if (
                not refresh
                and self._models_cache is not None
                and now - self._models_cache[0] < 60
            ):
                return list(self._models_cache[1])
            try:
                stdout, stderr, returncode = self._run_control_command(
                    [*self.config.extra_args, "--list-models"],
                    timeout_seconds=min(self.config.timeout_seconds, 30.0),
                )
            except (OSError, RuntimeError) as exc:
                raise CursorUnavailableError(f"Cursor model discovery failed: {exc}") from exc
            models = parse_model_catalog(stdout) if returncode == 0 else []
            if not models:
                detail = _redact(stderr, (self.config.cursor_api_key,))
                raise CursorUnavailableError(
                    f"Cursor model discovery returned no usable models: {detail}".rstrip()
                )
            self._models_cache = (now, models)
            return list(models)

    def cursor_version(self) -> str:
        stdout, stderr, returncode = self._run_control_command(
            [*self.config.extra_args, "--version"],
            timeout_seconds=min(self.config.timeout_seconds, 10.0),
        )
        if returncode != 0 or not stdout.strip():
            detail = _redact(stderr, (self.config.cursor_api_key,))
            raise CursorUnavailableError(f"Cursor version check failed: {detail}".rstrip())
        return stdout.strip().splitlines()[0]

    def diagnostics(self) -> dict[str, Any]:
        version = self.cursor_version()
        self._preflight_auth()
        models = self.list_models()
        return {
            "installed": True,
            "authenticated": True,
            "version": version,
            "protocol": "stream-json-2026-09",
            "models_available": bool(models),
            "model_count": len(models),
            "mode": self.config.mode,
            "workspace_policy": (
                "fresh-temporary-per-request"
                if self.config.mode == "hermes"
                else ("operator-selected" if self.config.workspace else "temporary")
            ),
        }

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if cancel_event is not None and cancel_event.is_set():
            raise CursorCancelledError("Cursor request was cancelled")
        self._preflight_auth()
        requested_model = model or "auto"
        if requested_model != "auto":
            available = self.list_models()
            by_lower = {candidate.lower(): candidate for candidate in available}
            if requested_model.lower() not in by_lower:
                raise CursorModelUnavailableError(
                    f"Cursor model is unavailable: {requested_model}"
                )
            requested_model = by_lower[requested_model.lower()]
        prompt = format_messages_as_prompt(messages, model=model, tools=tools, tool_choice=tool_choice)
        with self._request_workspace() as workspace:
            argv = build_cursor_argv(
                command=self._command,
                extra_args=self.config.extra_args,
                model=requested_model,
                workspace=workspace,
                mode=self.config.mode,
            )
            process: subprocess.Popen[Any] | None = None
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=workspace,
                    env=self._env(),
                    start_new_session=os.name == "posix",
                )
                with self._active_lock:
                    self._active.add(process)
                stdout, stderr, returncode = self._communicate_bounded(process, prompt, cancel_event)
            except FileNotFoundError as exc:
                raise CursorUnavailableError(
                    f"Cursor CLI command '{self.config.command}' was not found; install cursor-agent first"
                ) from exc
            finally:
                if process is not None:
                    self._terminate_process_tree(process)
                    with self._active_lock:
                        self._active.discard(process)

        accumulator = CursorProtocolAdapter(reject_native_tools=self.config.mode == "hermes")
        malformed_lines: list[str] = []
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                malformed_lines.append(line)
                continue
            if isinstance(event, dict):
                accumulator.feed(event)

        if returncode != 0 and not accumulator.is_error:
            detail = _redact(stderr or "\n".join(malformed_lines), (self.config.cursor_api_key,))
            raise CursorUnavailableError(
                f"Cursor CLI exited with status {returncode}: {detail}".rstrip()
            )
        if malformed_lines:
            raise CursorProtocolError("Cursor stream contained malformed NDJSON")
        allowed_tool_names = {
            function["name"]
            for tool in tools or []
            if isinstance(tool, dict)
            and isinstance((function := tool.get("function")), dict)
            and isinstance(function.get("name"), str)
        }
        try:
            return accumulator.to_completion(
                model=requested_model,
                prompt_tokens=estimate_prompt_tokens(messages, tools),
                allowed_tool_names=allowed_tool_names,
                tools=tools,
                tool_choice=tool_choice,
            )
        except (RuntimeError, CursorProtocolError) as exc:
            if isinstance(exc, CursorProtocolError):
                raise
            raise CursorRunnerError(_redact(str(exc), (self.config.cursor_api_key,))) from None
