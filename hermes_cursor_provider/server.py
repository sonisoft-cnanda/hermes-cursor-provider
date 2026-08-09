"""Authenticated loopback OpenAI-compatible HTTP bridge."""

from __future__ import annotations

import hmac
import json
import logging
import select
import socket
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .installer import _validate_token

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BridgeResponse:
    status: int
    body: bytes
    content_type: str = "application/json"


def _json_response(payload: dict[str, Any], status: int = 200) -> BridgeResponse:
    return BridgeResponse(
        status=status,
        body=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )


def _error(message: str, *, status: int, error_type: str) -> BridgeResponse:
    return _json_response(
        {"error": {"message": message, "type": error_type, "param": None, "code": None}},
        status=status,
    )


def validate_bind_host(host: str) -> str:
    candidate = host.strip()
    if candidate != "127.0.0.1":
        raise ValueError("Cursor bridge bind host must be exactly 127.0.0.1")
    return candidate


def encode_sse(completion: dict[str, Any]) -> bytes:
    """Convert one completed response into OpenAI-compatible SSE chunks."""
    choice = completion["choices"][0]
    message = choice.get("message") or {}
    common = {
        "id": completion["id"],
        "object": "chat.completion.chunk",
        "created": completion.get("created", 0),
        "model": completion.get("model", "auto"),
    }
    chunks: list[dict[str, Any]] = [
        {
            **common,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    ]
    delta: dict[str, Any] = {}
    if message.get("content") is not None:
        delta["content"] = message["content"]
    if message.get("reasoning_content"):
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {"index": index, **tool_call}
            for index, tool_call in enumerate(message["tool_calls"])
        ]
    if delta:
        chunks.append(
            {
                **common,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )
    chunks.append(
        {
            **common,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": choice.get("finish_reason") or "stop",
                }
            ],
            "usage": completion.get("usage"),
        }
    )
    frames = [
        "data: " + json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        for chunk in chunks
    ]
    frames.append("data: [DONE]")
    return ("\n\n".join(frames) + "\n\n").encode("utf-8")


class BridgeApplication:
    """Pure request handler shared by tests and the HTTP adapter."""

    def __init__(
        self,
        *,
        runner: Any,
        token: str,
        max_body_bytes: int = 8 * 1024 * 1024,
        max_concurrency: int = 1,
    ) -> None:
        token = _validate_token(token)
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.runner = runner
        self.token = token
        self.max_body_bytes = max_body_bytes
        self.max_concurrency = max_concurrency
        self._slots = threading.BoundedSemaphore(max_concurrency)

    def is_authorized(self, headers: dict[str, str]) -> bool:
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}
        supplied = normalized.get("authorization", "")
        expected = f"Bearer {self.token}"
        return hmac.compare_digest(supplied, expected)

    def try_acquire_slot(self) -> bool:
        return self._slots.acquire(blocking=False)

    def release_slot(self) -> None:
        self._slots.release()

    def handle(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        *,
        cancel_event: threading.Event | None = None,
        preauthorized: bool = False,
        admitted: bool = False,
    ) -> BridgeResponse:
        normalized_headers = {str(key).lower(): str(value) for key, value in headers.items()}
        if not preauthorized and not self.is_authorized(normalized_headers):
            return _error("Invalid or missing bearer token", status=401, error_type="authentication_error")
        if len(body) > self.max_body_bytes:
            return _error("Request body is too large", status=413, error_type="invalid_request_error")

        route = urlsplit(path).path.rstrip("/") or "/"
        method = method.upper()
        if method == "GET" and route == "/health":
            return _json_response({"status": "ok"})
        if method == "GET" and route == "/v1/models":
            acquired_here = not admitted
            if acquired_here and not self.try_acquire_slot():
                return _error("Cursor bridge is busy", status=429, error_type="rate_limit_error")
            try:
                models = self.runner.list_models()
                return _json_response(
                    {
                        "object": "list",
                        "data": [
                            {"id": model, "object": "model", "owned_by": "cursor"}
                            for model in models
                        ],
                    }
                )
            finally:
                if acquired_here:
                    self.release_slot()
        if not (method == "POST" and route == "/v1/chat/completions"):
            return _error("Route not found", status=404, error_type="invalid_request_error")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return _error("Request body must be valid UTF-8 JSON", status=400, error_type="invalid_request_error")
        if not isinstance(payload, dict):
            return _error("Request body must be a JSON object", status=400, error_type="invalid_request_error")
        model = payload.get("model")
        messages = payload.get("messages")
        tools = payload.get("tools")
        if not isinstance(model, str) or not model.strip():
            return _error("model must be a non-empty string", status=400, error_type="invalid_request_error")
        if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
            return _error("messages must be an array of objects", status=400, error_type="invalid_request_error")
        if tools is not None and not isinstance(tools, list):
            return _error("tools must be an array when provided", status=400, error_type="invalid_request_error")

        acquired_here = not admitted
        if acquired_here and not self.try_acquire_slot():
            return _error("Cursor bridge is busy", status=429, error_type="rate_limit_error")
        try:
            completion_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": payload.get("tool_choice"),
            }
            if cancel_event is not None:
                completion_kwargs["cancel_event"] = cancel_event
            completion = self.runner.complete(**completion_kwargs)
        except Exception as exc:  # noqa: BLE001 - HTTP boundary normalizes runner failures
            logger.warning("Cursor completion failed (%s)", type(exc).__name__)
            return _error("Cursor completion failed", status=502, error_type="cursor_error")
        finally:
            if acquired_here:
                self.release_slot()

        if payload.get("stream") is True:
            return BridgeResponse(status=200, body=encode_sse(completion), content_type="text/event-stream")
        return _json_response(completion)


class _BridgeHandler(BaseHTTPRequestHandler):
    server_version = "HermesCursorProvider/0.1"

    @property
    def app(self) -> BridgeApplication:
        return self.server.bridge_app  # type: ignore[attr-defined]

    def _dispatch(self) -> None:
        headers = dict(self.headers.items())
        if not self.app.is_authorized(headers):
            self._send(
                _error(
                    "Invalid or missing bearer token",
                    status=401,
                    error_type="authentication_error",
                )
            )
            return

        admitted = False
        if self.command.upper() == "POST":
            admitted = self.app.try_acquire_slot()
            if not admitted:
                self._send(_error("Cursor bridge is busy", status=429, error_type="rate_limit_error"))
                return

        try:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                content_length = int(raw_length)
            except ValueError:
                self._send(_error("Invalid Content-Length", status=400, error_type="invalid_request_error"))
                return
            if content_length < 0:
                self._send(_error("Invalid Content-Length", status=400, error_type="invalid_request_error"))
                return
            if content_length > self.app.max_body_bytes:
                self._send(_error("Request body is too large", status=413, error_type="invalid_request_error"))
                return
            try:
                body = self.rfile.read(content_length) if content_length else b""
            except (OSError, TimeoutError):
                self._send(_error("Request body timed out", status=408, error_type="request_timeout"))
                return
            self._dispatch_authenticated(headers, body, admitted=admitted)
        finally:
            if admitted:
                self.app.release_slot()

    def _dispatch_authenticated(
        self,
        headers: dict[str, str],
        body: bytes,
        *,
        admitted: bool,
    ) -> None:
        cancel_event = threading.Event()
        finished = threading.Event()

        def watch_disconnect() -> None:
            while not finished.wait(0.05):
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                        cancel_event.set()
                        return
                except (OSError, ValueError):
                    cancel_event.set()
                    return

        monitor = threading.Thread(target=watch_disconnect, daemon=True)
        monitor.start()
        try:
            response = self.app.handle(
                self.command,
                self.path,
                headers,
                body,
                cancel_event=cancel_event,
                preauthorized=True,
                admitted=admitted,
            )
        finally:
            finished.set()
            monitor.join(timeout=0.2)
        self._send(response)

    def _send(self, response: BridgeResponse) -> None:
        try:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response.body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def log_message(self, format: str, *args: object) -> None:
        logger.info("bridge http: " + format, *args)


class CursorBridgeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        app: BridgeApplication,
        *,
        client_timeout: float = 10.0,
        max_handler_threads: int = 16,
    ) -> None:
        if client_timeout <= 0:
            raise ValueError("client_timeout must be positive")
        if max_handler_threads <= 0:
            raise ValueError("max_handler_threads must be positive")
        self.bridge_app = app
        self._client_timeout = client_timeout
        self._handler_slots = threading.BoundedSemaphore(max_handler_threads)
        super().__init__(server_address, _BridgeHandler)

    def get_request(self) -> tuple[socket.socket, Any]:
        request, client_address = super().get_request()
        request.settimeout(self._client_timeout)
        return request, client_address

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._handler_slots.acquire(blocking=False):
            response = _error("Cursor bridge is busy", status=429, error_type="rate_limit_error")
            payload = (
                b"HTTP/1.1 429 Too Many Requests\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(response.body)}\r\n".encode()
                + b"Cache-Control: no-store\r\nConnection: close\r\n\r\n"
                + response.body
            )
            try:
                request.sendall(payload)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._handler_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_slots.release()


def create_http_server(
    *,
    host: str,
    port: int,
    app: BridgeApplication,
    client_timeout: float = 10.0,
    max_handler_threads: int = 16,
) -> CursorBridgeHTTPServer:
    if not (0 <= port <= 65535):
        raise ValueError("port must be between 0 and 65535")
    validated_host = validate_bind_host(host)
    return CursorBridgeHTTPServer(
        (validated_host, port),
        app,
        client_timeout=client_timeout,
        max_handler_threads=max_handler_threads,
    )
