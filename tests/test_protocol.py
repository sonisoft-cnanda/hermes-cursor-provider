from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cursor_provider.protocol import (
    CursorStreamAccumulator,
    extract_tool_calls,
    format_messages_as_prompt,
)
from hermes_cursor_provider.runner import (
    CursorRunner,
    RunnerConfig,
    build_cursor_argv,
    parse_model_catalog,
)


def _event(**fields: object) -> str:
    return json.dumps(fields)


SUCCESS_EVENTS = [
    {"type": "system", "subtype": "init", "session_id": "s-1", "model": "Auto"},
    {"type": "thinking", "subtype": "delta", "text": "thinking-bit", "session_id": "s-1"},
    {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]},
        "session_id": "s-1",
    },
    {
        "type": "result",
        "subtype": "success",
        "duration_ms": 1234,
        "is_error": False,
        "result": "Hello world",
        "session_id": "s-1",
        "request_id": "r-1",
        "usage": {"inputTokens": 42, "outputTokens": 13, "cacheReadTokens": 9},
    },
]


def test_stream_accumulator_builds_openai_completion() -> None:
    accumulator = CursorStreamAccumulator()
    for event in SUCCESS_EVENTS:
        accumulator.feed(event)

    completion = accumulator.to_completion(model="composer-2.5", prompt_tokens=7)

    assert completion["id"] == "r-1"
    assert completion["model"] == "composer-2.5"
    assert completion["choices"][0]["message"]["content"] == "Hello world"
    assert completion["choices"][0]["message"]["reasoning_content"] == "thinking-bit"
    assert completion["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 13,
        "total_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 7},
    }


def test_stream_accumulator_raises_cursor_error() -> None:
    accumulator = CursorStreamAccumulator()
    accumulator.feed(
        {
            "type": "result",
            "subtype": "error",
            "is_error": True,
            "result": "quota exceeded",
            "request_id": "r-2",
        }
    )

    with pytest.raises(RuntimeError, match="quota exceeded"):
        accumulator.to_completion(model="auto", prompt_tokens=1)


def test_unknown_stream_events_are_ignored() -> None:
    accumulator = CursorStreamAccumulator()
    accumulator.feed({"type": "future-event", "payload": 1})
    assert accumulator.terminal is False
    assert accumulator.text == ""


def test_cached_tokens_never_exceed_reported_prompt_tokens() -> None:
    accumulator = CursorStreamAccumulator()
    accumulator.feed(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "ok",
            "usage": {"inputTokens": 1, "outputTokens": 1, "cacheReadTokens": 99},
        }
    )

    completion = accumulator.to_completion(model="auto", prompt_tokens=7)

    assert completion["usage"]["prompt_tokens_details"]["cached_tokens"] == 7


def test_format_prompt_preserves_transcript_and_tool_schema() -> None:
    prompt = format_messages_as_prompt(
        messages=[
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
            {"role": "user", "content": "again"},
        ],
        model="composer-2.5",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "kanban",
                    "description": "manage tasks",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert prompt.index("ping") < prompt.index("pong") < prompt.index("again")
    assert "Hermes-side tools" in prompt
    assert "<tool_call>" in prompt
    assert '"kanban"' in prompt


def test_toolless_prompt_uses_short_auxiliary_contract() -> None:
    prompt = format_messages_as_prompt(
        messages=[{"role": "user", "content": "title this"}],
        tools=None,
    )
    assert "auxiliary call" in prompt.lower()
    assert "do not run" in prompt.lower()
    assert "<tool_call>" not in prompt


def test_extract_tool_calls_removes_blocks_and_normalizes_arguments() -> None:
    text = (
        "Checking.\n"
        '<tool_call>{"id":"call_1","type":"function","function":'
        '{"name":"kanban","arguments":{"action":"list"}}}</tool_call>\n'
    )

    content, calls = extract_tool_calls(text)

    assert content == "Checking."
    assert calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "kanban", "arguments": '{"action":"list"}'},
        }
    ]


def test_extract_tool_calls_accepts_live_cursor_simplified_shape() -> None:
    text = (
        "<tool_call>\n"
        '{"name": "hermes_probe", "arguments": "{\\"value\\": \\"ok\\"}"}\n'
        "</tool_call>"
    )

    content, calls = extract_tool_calls(text)

    assert content == ""
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "hermes_probe"
    assert json.loads(calls[0]["function"]["arguments"]) == {"value": "ok"}


def test_stream_tool_calls_are_limited_to_tools_offered_by_hermes() -> None:
    accumulator = CursorStreamAccumulator()
    accumulator.feed(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": '<tool_call>{"name":"dangerous_existing_tool","arguments":{}}</tool_call>',
        }
    )

    blocked = accumulator.to_completion(
        model="auto",
        prompt_tokens=1,
        allowed_tool_names=frozenset(),
    )
    allowed = accumulator.to_completion(
        model="auto",
        prompt_tokens=1,
        allowed_tool_names=frozenset({"dangerous_existing_tool"}),
    )

    assert blocked["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in blocked["choices"][0]["message"]
    assert allowed["choices"][0]["finish_reason"] == "tool_calls"
    assert allowed["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "dangerous_existing_tool"


def test_agent_mode_omits_mode_and_secret_from_argv() -> None:
    argv = build_cursor_argv(
        command="cursor-agent",
        extra_args=(),
        model="composer-2.5",
        workspace="/tmp/work",
        mode="agent",
    )
    assert argv[0] == "cursor-agent"
    assert "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--model") + 1] == "composer-2.5"
    assert argv[argv.index("--workspace") + 1] == "/tmp/work"
    assert "--mode" not in argv
    assert "--api-key" not in argv
    assert "--force" in argv
    assert "--trust" in argv


def test_read_only_modes_are_explicit_and_not_forced() -> None:
    for mode in ("ask", "plan"):
        argv = build_cursor_argv(
            command="cursor-agent",
            extra_args=(),
            model="auto",
            workspace="/tmp/work",
            mode=mode,
        )
        assert argv[argv.index("--mode") + 1] == mode
        assert "--trust" in argv
        assert "--force" not in argv


def test_runner_default_mode_is_read_only_ask() -> None:
    assert RunnerConfig().mode == "ask"


def test_runner_resolves_cursor_command_to_absolute_executable() -> None:
    runner = CursorRunner(RunnerConfig(command="python3"))
    assert Path(runner._command).is_absolute()
    assert Path(runner._command).is_file()


def test_configured_workspace_must_already_exist(tmp_path: Path) -> None:
    missing = tmp_path / "typo-workspace"
    runner = CursorRunner(
        RunnerConfig(
            command=sys.executable,
            workspace=str(missing),
            cursor_api_key="cursor-test-key",
        )
    )
    with pytest.raises(ValueError, match="existing directory"):
        runner._workspace()
    assert not missing.exists()


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        build_cursor_argv(
            command="cursor-agent",
            extra_args=(),
            model="auto",
            workspace="/tmp/work",
            mode="unsafe-mystery",
        )


@pytest.mark.parametrize(
    "argument",
    (
        "--api-key",
        "--api-key=secret",
        "--force",
        "--trust",
        "--mode=agent",
        "--workspace=/tmp/escape",
        "--model=other",
        "--output-format=text",
        "--sandbox=disabled",
        "--resume=session",
        "-p",
    ),
)
def test_extra_args_cannot_override_security_or_protocol_flags(argument: str) -> None:
    with pytest.raises(ValueError, match="extra argument"):
        RunnerConfig(extra_args=(argument,))


def test_parse_model_catalog_prioritizes_without_dropping_models() -> None:
    output = "\n".join(
        [
            "Available models",
            "z-random - Random",
            "composer-2.5-fast - Composer Fast",
            "auto - Auto",
            "composer-2.5 - Composer",
            "z-random - duplicate",
            "Tip: use --model <id>",
        ]
    )
    assert parse_model_catalog(output) == ["auto", "composer-2.5", "composer-2.5-fast", "z-random"]


def test_runner_executes_real_fake_process_and_keeps_key_out_of_argv(tmp_path: Path) -> None:
    fake = tmp_path / "fake_cursor.py"
    fake.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "prompt = sys.stdin.read()",
                "assert 'hello from test' in prompt",
                "assert os.environ.get('CURSOR_API_KEY') == 'crsr_test_secret'",
                "assert 'crsr_test_secret' not in sys.argv",
                "print(json.dumps({'type':'assistant','message':{'content':[{'type':'text','text':'fake ok'}]}}))",
                "print(json.dumps({'type':'result','subtype':'success','is_error':False,'result':'fake ok','request_id':'fake-r','usage':{'inputTokens':3,'outputTokens':2,'cacheReadTokens':0}}))",
            ]
        )
    )
    config = RunnerConfig(
        command=sys.executable,
        extra_args=(str(fake),),
        mode="ask",
        timeout_seconds=5,
        cursor_api_key="crsr_test_secret",
    )

    with CursorRunner(config) as runner:
        completion = runner.complete(
            model="auto",
            messages=[{"role": "user", "content": "hello from test"}],
            tools=None,
            tool_choice=None,
        )

    assert completion["choices"][0]["message"]["content"] == "fake ok"
    assert completion["id"] == "fake-r"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_timeout_terminates_cursor_process_group(tmp_path: Path) -> None:
    script = tmp_path / "fake_cursor_tree.py"
    pid_file = tmp_path / "child.pid"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script.write_text(
        "\n".join(
            [
                "import pathlib, subprocess, sys, time",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])",
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid))",
                "time.sleep(60)",
            ]
        )
    )
    runner = CursorRunner(
        RunnerConfig(
            command=sys.executable,
            extra_args=(str(script), str(pid_file)),
            mode="ask",
            workspace=str(workspace),
            timeout_seconds=0.2,
            cursor_api_key="test-secret",
        )
    )

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    child_pid = 0
    try:
        with pytest.raises(RuntimeError, match="request timeout"):
            runner.complete(
                model="auto",
                messages=[{"role": "user", "content": "timeout probe"}],
                tools=None,
                tool_choice=None,
            )
        child_pid = int(pid_file.read_text())
        deadline = time.monotonic() + 2
        while alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not alive(child_pid)
    finally:
        runner.close()
        if child_pid and alive(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_cursor_subprocess_environment_is_strictly_allowlisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NOUS_API_KEY", "nous-sentinel")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-sentinel")
    monkeypatch.setenv("SOLANA_WALLET_SECRET", "wallet-sentinel")
    monkeypatch.setenv("PYTHONPATH", "/tmp/injected-pythonpath")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected-preload.so")
    monkeypatch.setenv("CURSOR_BRIDGE_API_KEY", "bridge-sentinel")
    runner = CursorRunner(
        RunnerConfig(
            command=sys.executable,
            workspace=str(tmp_path),
            cursor_api_key="cursor-upstream-secret",
        )
    )

    child_env = runner._env()

    assert child_env["PATH"] == os.environ["PATH"]
    assert child_env["HOME"] == str(tmp_path)
    assert child_env["CURSOR_API_KEY"] == "cursor-upstream-secret"
    for blocked in (
        "NOUS_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "SOLANA_WALLET_SECRET",
        "PYTHONPATH",
        "LD_PRELOAD",
        "CURSOR_BRIDGE_API_KEY",
    ):
        assert blocked not in child_env
    runner.close()


def test_cursor_error_never_exposes_upstream_key(tmp_path: Path) -> None:
    fake = tmp_path / "fake_cursor_error.py"
    fake.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "sys.stdin.read()",
                "secret = os.environ['CURSOR_API_KEY']",
                "print(json.dumps({'type':'result','subtype':'error','is_error':True,'result':'failure ' + secret}))",
            ]
        )
    )
    secret = "cursor-upstream-secret-sentinel"
    runner = CursorRunner(
        RunnerConfig(
            command=sys.executable,
            extra_args=(str(fake),),
            workspace=str(tmp_path),
            timeout_seconds=5,
            cursor_api_key=secret,
        )
    )

    with pytest.raises(RuntimeError) as captured:
        runner.complete(
            model="auto",
            messages=[{"role": "user", "content": "error probe"}],
            tools=None,
            tool_choice=None,
        )

    assert secret not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)
    runner.close()


def test_cursor_output_limit_terminates_noisy_process(tmp_path: Path) -> None:
    fake = tmp_path / "fake_cursor_noisy.py"
    fake.write_text(
        "\n".join(
            [
                "import sys, time",
                "sys.stdin.read()",
                "sys.stdout.write('x' * 65536)",
                "sys.stdout.flush()",
                "time.sleep(30)",
            ]
        )
    )
    runner = CursorRunner(
        RunnerConfig(
            command=sys.executable,
            extra_args=(str(fake),),
            workspace=str(tmp_path),
            timeout_seconds=5,
            max_output_bytes=1024,
            cursor_api_key="cursor-test-key",
        )
    )

    with pytest.raises(RuntimeError, match="output limit"):
        runner.complete(
            model="auto",
            messages=[{"role": "user", "content": "noise probe"}],
            tools=None,
            tool_choice=None,
        )
    runner.close()


def test_cursor_cancellation_terminates_request_process(tmp_path: Path) -> None:
    fake = tmp_path / "fake_cursor_cancel.py"
    pid_file = tmp_path / "cancel.pid"
    fake.write_text(
        "\n".join(
            [
                "import os, sys, time",
                "open(sys.argv[1], 'w').write(str(os.getpid()))",
                "sys.stdin.read()",
                "time.sleep(30)",
            ]
        )
    )
    cancel = threading.Event()
    runner = CursorRunner(
        RunnerConfig(
            command=sys.executable,
            extra_args=(str(fake), str(pid_file)),
            workspace=str(tmp_path),
            timeout_seconds=5,
            cursor_api_key="cursor-test-key",
        )
    )
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            runner.complete(
                model="auto",
                messages=[{"role": "user", "content": "cancel probe"}],
                tools=None,
                tool_choice=None,
                cancel_event=cancel,
            )
    finally:
        timer.cancel()
        runner.close()

    if pid_file.exists():
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
