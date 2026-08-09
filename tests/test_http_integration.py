from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from hermes_cursor_provider.installer import install_plugin
from hermes_cursor_provider.runner import CursorRunner, RunnerConfig
from hermes_cursor_provider.server import BridgeApplication, create_http_server

TEST_TOKEN = "integration-test-token-" + "i" * 32


class IntegrationRunner:
    def list_models(self) -> list[str]:
        return ["auto", "composer-2.5"]

    def complete(self, *, model, messages, tools, tool_choice, cancel_event=None):
        return {
            "id": "integration-1",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "socket ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }


@pytest.fixture
def running_bridge():
    app = BridgeApplication(runner=IntegrationRunner(), token=TEST_TOKEN)
    server = create_http_server(host="127.0.0.1", port=0, app=app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_http_socket_enforces_auth_and_serves_models(running_bridge: str) -> None:
    unauthorized = urllib.request.Request(f"{running_bridge}/v1/models")
    with pytest.raises(urllib.error.HTTPError) as captured:
        urllib.request.urlopen(unauthorized, timeout=2)
    assert captured.value.code == 401

    authorized = urllib.request.Request(
        f"{running_bridge}/v1/models",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    with urllib.request.urlopen(authorized, timeout=2) as response:
        payload = json.loads(response.read())
    assert [item["id"] for item in payload["data"]] == ["auto", "composer-2.5"]


def test_openai_sdk_can_consume_bridge(running_bridge: str) -> None:
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(api_key=TEST_TOKEN, base_url=f"{running_bridge}/v1")

    completion = client.chat.completions.create(
        model="composer-2.5",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert completion.id == "integration-1"
    assert completion.choices[0].message.content == "socket ok"


def test_installed_shim_registers_with_public_provider_api(tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes-home"
    install_plugin(hermes_home, token=TEST_TOKEN)
    fake_core = tmp_path / "fake-core"
    providers = fake_core / "providers"
    providers.mkdir(parents=True)
    (providers / "__init__.py").write_text(
        "REGISTRY = {}\n"
        "def register_provider(profile): REGISTRY[profile.name] = profile\n"
    )
    (providers / "base.py").write_text(
        "class ProviderProfile:\n"
        "    def __init__(self, **kwargs): self.__dict__.update(kwargs)\n"
    )
    script = "\n".join(
        [
            "import importlib.util, json, pathlib, providers",
            f"path = pathlib.Path({str(hermes_home)!r}) / 'plugins/model-providers/cursor/__init__.py'",
            "spec = importlib.util.spec_from_file_location('cursor_plugin', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "profile = providers.REGISTRY['cursor']",
            "print(json.dumps({'name': profile.name, 'base_url': profile.base_url, 'auth_type': profile.auth_type, 'env_vars': profile.env_vars}))",
        ]
    )
    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([str(fake_core), project_root])

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    profile = json.loads(completed.stdout)
    assert profile == {
        "name": "cursor",
        "base_url": "http://127.0.0.1:8765/v1",
        "auth_type": "api_key",
        "env_vars": ["CURSOR_BRIDGE_API_KEY"],
    }


def test_full_stack_openai_to_fake_cursor_tool_call(tmp_path: Path) -> None:
    openai = pytest.importorskip("openai")
    fake = tmp_path / "fake_cursor.py"
    fake.write_text(
        "\n".join(
            [
                "import json, os, sys",
                "prompt = sys.stdin.read()",
                "assert 'Hermes-side tools' in prompt",
                "assert os.environ.get('CURSOR_API_KEY') == 'cursor-secret'",
                "assert 'cursor-secret' not in sys.argv",
                "text = 'Checking.\\n<tool_call>{\"id\":\"call_9\",\"type\":\"function\",\"function\":{\"name\":\"kanban\",\"arguments\":{\"action\":\"list\"}}}</tool_call>'",
                "print(json.dumps({'type':'assistant','message':{'content':[{'type':'text','text':text}]}}))",
                "print(json.dumps({'type':'result','subtype':'success','is_error':False,'result':text,'request_id':'full-stack','usage':{'inputTokens':5,'outputTokens':3,'cacheReadTokens':1}}))",
            ]
        )
    )
    runner = CursorRunner(
        RunnerConfig(
            command=sys.executable,
            extra_args=(str(fake),),
            mode="ask",
            timeout_seconds=5,
            cursor_api_key="cursor-secret",
        )
    )
    app = BridgeApplication(runner=runner, token=TEST_TOKEN)
    server = create_http_server(host="127.0.0.1", port=0, app=app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        client = openai.OpenAI(api_key=TEST_TOKEN, base_url=f"http://{host}:{port}/v1")
        completion = client.chat.completions.create(
            model="auto",
            messages=[{"role": "user", "content": "list tasks"}],
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
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        runner.close()

    assert completion.id == "full-stack"
    assert completion.choices[0].finish_reason == "tool_calls"
    tool_call = completion.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "kanban"
    assert json.loads(tool_call.function.arguments) == {"action": "list"}
