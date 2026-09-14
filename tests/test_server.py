from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from hermes_cursor_provider.protocol import CursorProtocolError
from hermes_cursor_provider.runner import (
    CursorAuthenticationError,
    CursorModelUnavailableError,
    CursorTimeoutError,
)
from hermes_cursor_provider.server import (
    BridgeApplication,
    BridgeResponse,
    create_http_server,
    encode_sse,
    validate_bind_host,
)

TEST_TOKEN = "bridge-test-token-" + "x" * 32


class StubRunner:
    class Config:
        mode = "hermes"

    config = Config()

    def diagnostics(self) -> dict[str, object]:
        return {
            "installed": True,
            "authenticated": True,
            "version": "test",
            "models_available": True,
            "model_count": 2,
        }

    def list_models(self) -> list[str]:
        return ["auto", "composer-2.5"]

    def complete(self, *, model, messages, tools, tool_choice, cancel_event=None):
        del cancel_event
        return {
            "id": "stub-1",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "stub response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        }


def _auth(token: str = TEST_TOKEN) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def test_bridge_rejects_weak_bearer_token() -> None:
    with pytest.raises(ValueError, match="43"):
        BridgeApplication(runner=StubRunner(), token="too-short")

    with pytest.raises(ValueError, match="weak"):
        BridgeApplication(runner=StubRunner(), token="a" * 43)


def test_bridge_requires_bearer_token_for_all_routes() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    response = app.handle("GET", "/health", {}, b"")
    assert response.status == 401
    assert json.loads(response.body)["error"]["type"] == "authentication_error"


def test_health_and_models_are_openai_shaped() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)

    health = app.handle("GET", "/health", _auth(), b"")
    models = app.handle("GET", "/v1/models", _auth(), b"")

    assert health.status == 200
    health_payload = json.loads(health.body)
    assert health_payload["status"] == "ok"
    assert health_payload["mode"] == "hermes"
    assert health_payload["cursor"]["version"] == "test"
    assert health_payload["capabilities"]["streaming"] is False
    assert json.loads(models.body) == {
        "object": "list",
        "data": [
            {"id": "auto", "object": "model", "owned_by": "cursor"},
            {"id": "composer-2.5", "object": "model", "owned_by": "cursor"},
        ],
    }


def test_non_streaming_chat_completion() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    body = json.dumps(
        {
            "model": "composer-2.5",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }
    ).encode()

    response = app.handle("POST", "/v1/chat/completions", _auth(), body)

    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["choices"][0]["message"]["content"] == "stub response"


def test_streaming_chat_completion_is_valid_sse() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    body = json.dumps(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
    ).encode()

    response = app.handle("POST", "/v1/chat/completions", _auth(), body)

    assert response.status == 200
    assert response.content_type == "text/event-stream"
    frames = response.body.decode().strip().split("\n\n")
    assert frames[-1] == "data: [DONE]"
    payloads = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
    assert any(chunk["choices"][0]["delta"].get("content") == "stub response" for chunk in payloads)
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_invalid_json_returns_openai_error() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    response = app.handle("POST", "/v1/chat/completions", _auth(), b"not-json")
    assert response.status == 400
    assert json.loads(response.body)["error"]["type"] == "invalid_request_error"


def test_missing_messages_returns_openai_error() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    response = app.handle(
        "POST",
        "/v1/chat/completions",
        _auth(),
        json.dumps({"model": "auto"}).encode(),
    )
    assert response.status == 400


def test_cursor_failure_does_not_expose_exception_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "cursor-upstream-secret-sentinel"

    class SecretFailureRunner(StubRunner):
        def complete(self, **_: object) -> dict[str, object]:
            raise RuntimeError(f"upstream failed with {secret}")

    app = BridgeApplication(runner=SecretFailureRunner(), token=TEST_TOKEN)
    body = json.dumps(
        {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()

    response = app.handle("POST", "/v1/chat/completions", _auth(), body)

    assert response.status == 502
    assert secret.encode() not in response.body
    assert secret not in caplog.text
    assert json.loads(response.body)["error"]["type"] == "cursor_error"


@pytest.mark.parametrize(
    ("failure", "status", "error_type"),
    [
        (CursorModelUnavailableError("missing"), 404, "model_not_found"),
        (CursorAuthenticationError("login"), 503, "provider_unavailable"),
        (CursorTimeoutError("slow"), 504, "timeout_error"),
        (CursorProtocolError("bad event"), 502, "cursor_protocol_error"),
    ],
)
def test_typed_cursor_failures_map_to_safe_openai_errors(
    failure: Exception,
    status: int,
    error_type: str,
) -> None:
    class FailureRunner(StubRunner):
        def complete(self, **_: object) -> dict[str, object]:
            raise failure

    app = BridgeApplication(runner=FailureRunner(), token=TEST_TOKEN)
    body = json.dumps(
        {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()

    response = app.handle("POST", "/v1/chat/completions", _auth(), body)

    assert response.status == status
    assert json.loads(response.body)["error"]["type"] == error_type


def test_oversized_request_is_rejected_before_json_parse() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN, max_body_bytes=8)
    response = app.handle("POST", "/v1/chat/completions", _auth(), b"{" + b"x" * 20)
    assert response.status == 413


def test_only_loopback_hosts_are_allowed_by_default() -> None:
    assert validate_bind_host("127.0.0.1") == "127.0.0.1"
    for forbidden in ("localhost", "::1", "0.0.0.0", "::", "192.168.1.4"):
        with pytest.raises(ValueError, match="127.0.0.1"):
            validate_bind_host(forbidden)


def test_concurrency_is_bounded_and_busy_requests_get_429() -> None:
    class BlockingRunner(StubRunner):
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def complete(self, *, model, messages, tools, tool_choice, cancel_event=None):
            del cancel_event
            self.entered.set()
            assert self.release.wait(timeout=2)
            return super().complete(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

    runner = BlockingRunner()
    app = BridgeApplication(runner=runner, token=TEST_TOKEN, max_concurrency=1)
    body = json.dumps(
        {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    first: list[BridgeResponse] = []
    thread = threading.Thread(
        target=lambda: first.append(app.handle("POST", "/v1/chat/completions", _auth(), body))
    )
    thread.start()
    assert runner.entered.wait(timeout=2)

    busy = app.handle("POST", "/v1/chat/completions", _auth(), body)

    assert busy.status == 429
    assert json.loads(busy.body)["error"]["type"] == "rate_limit_error"
    runner.release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert first and first[0].status == 200


def test_http_rejects_unauthorized_request_before_reading_body() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    server = create_http_server(host="127.0.0.1", port=0, app=app, client_timeout=1.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = socket.create_connection((str(host), int(port)), timeout=2)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Authorization: Bearer invalid\r\n"
            b"Content-Length: 1000\r\n\r\n"
        )
        response = client.recv(4096)
        client.close()
        assert b" 401 " in response
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("method", ["HEAD", "OPTIONS", "PUT", "PATCH", "DELETE"])
def test_all_http_methods_require_authentication(method: str) -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    server = create_http_server(host="127.0.0.1", port=0, app=app, client_timeout=1.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = socket.create_connection((str(host), int(port)), timeout=2)
        client.sendall(
            f"{method} /unknown HTTP/1.0\r\nContent-Length: 999\r\n\r\n".encode()
        )
        response = client.recv(4096)
        client.close()
        assert b" 401 " in response
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_rejects_transfer_encoding_before_dispatch() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    server = create_http_server(host="127.0.0.1", port=0, app=app, client_timeout=1.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = socket.create_connection((str(host), int(port)), timeout=2)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
            + b"Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
        )
        response_parts = []
        while chunk := client.recv(4096):
            response_parts.append(chunk)
        client.close()
        response = b"".join(response_parts)
        assert b" 400 " in response
        assert b"Transfer-Encoding" in response
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_rejects_duplicate_content_length() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    server = create_http_server(host="127.0.0.1", port=0, app=app, client_timeout=1.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = socket.create_connection((str(host), int(port)), timeout=2)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.0\r\n"
            + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
            + b"Content-Length: 2\r\nContent-Length: 3\r\n\r\n{}"
        )
        response = client.recv(4096)
        client.close()
        assert b" 400 " in response
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_authenticated_slow_body_holds_only_one_admission_slot() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN, max_concurrency=1)
    server = create_http_server(host="127.0.0.1", port=0, app=app, client_timeout=1.0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    first: socket.socket | None = None
    second: socket.socket | None = None
    try:
        host, port = server.server_address[:2]
        first = socket.create_connection((str(host), int(port)), timeout=2)
        first.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
            + b"Content-Length: 1000\r\n\r\n"
        )
        time.sleep(0.05)

        second = socket.create_connection((str(host), int(port)), timeout=2)
        second.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
            + b"Content-Length: 2\r\n\r\n{}"
        )
        response = second.recv(4096)
        assert b" 429 " in response
    finally:
        if first is not None:
            first.close()
        if second is not None:
            second.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_slow_body_times_out_and_releases_admission_slot() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN, max_concurrency=1)
    server = create_http_server(host="127.0.0.1", port=0, app=app, client_timeout=0.1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = socket.create_connection((str(host), int(port)), timeout=2)
        client.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
            + b"Content-Length: 1000\r\n\r\n"
        )
        response = client.recv(4096)
        client.close()
        assert b" 408 " in response

        body = json.dumps(
            {"model": "auto", "messages": [{"role": "user", "content": "after timeout"}]}
        ).encode()
        follow_up = socket.create_connection((str(host), int(port)), timeout=2)
        follow_up.sendall(
            b"POST /v1/chat/completions HTTP/1.0\r\n"
            + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        follow_up_response = follow_up.recv(4096)
        follow_up.close()
        assert b" 200 " in follow_up_response
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_total_http_handler_threads_are_bounded() -> None:
    app = BridgeApplication(runner=StubRunner(), token=TEST_TOKEN)
    server = create_http_server(
        host="127.0.0.1",
        port=0,
        app=app,
        client_timeout=1.0,
        max_handler_threads=1,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    first: socket.socket | None = None
    second: socket.socket | None = None
    try:
        host, port = server.server_address[:2]
        first = socket.create_connection((str(host), int(port)), timeout=2)
        first.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n")
        time.sleep(0.05)

        second = socket.create_connection((str(host), int(port)), timeout=2)
        second.sendall(b"GET /health HTTP/1.0\r\n\r\n")
        response = second.recv(4096)
        assert b" 429 " in response
    finally:
        if first is not None:
            first.close()
        if second is not None:
            second.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_sse_encoder_preserves_tool_calls() -> None:
    completion = {
        "id": "tool-1",
        "object": "chat.completion",
        "created": 1,
        "model": "auto",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "kanban", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    encoded = encode_sse(completion).decode()
    assert '"tool_calls"' in encoded
    assert "data: [DONE]" in encoded


def test_http_disconnect_cancels_inflight_cursor_request() -> None:
    cancelled = threading.Event()

    class CancellationRunner(StubRunner):
        def complete(self, *, cancel_event=None, **_: object) -> dict[str, object]:
            assert cancel_event is not None
            assert cancel_event.wait(timeout=2)
            cancelled.set()
            raise RuntimeError("cancelled")

    app = BridgeApplication(runner=CancellationRunner(), token=TEST_TOKEN)
    server = create_http_server(host="127.0.0.1", port=0, app=app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    body = json.dumps(
        {"model": "auto", "messages": [{"role": "user", "content": "disconnect"}]}
    ).encode()
    request = (
        b"POST /v1/chat/completions HTTP/1.0\r\n"
        + f"Authorization: Bearer {TEST_TOKEN}\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    try:
        host, port = server.server_address[:2]
        client = socket.create_connection((str(host), int(port)), timeout=2)
        client.sendall(request)
        time.sleep(0.1)
        client.close()
        assert cancelled.wait(timeout=3)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
