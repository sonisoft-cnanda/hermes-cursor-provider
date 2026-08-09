"""Cursor stream-json and OpenAI message translation."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered: list[str] = []
        for part in content:
            if isinstance(part, str):
                rendered.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    rendered.append(part["text"])
                elif part.get("type") in {"image", "image_url", "input_image"}:
                    rendered.append("[image omitted from Cursor CLI bridge]")
                else:
                    rendered.append(json.dumps(part, ensure_ascii=False, sort_keys=True))
            else:
                rendered.append(str(part))
        return "\n".join(item for item in rendered if item)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    """Serialize an OpenAI transcript into one Cursor CLI prompt."""
    sections: list[str] = []
    if tools:
        sections.extend(
            [
                "You are powering a chat session inside Hermes Agent.",
                "You have built-in cursor-agent tools and the Hermes-side tools listed below.",
                "Use built-in cursor-agent tools for local shell and file work.",
                (
                    "To invoke a Hermes-side tool, emit exactly one OpenAI-shaped JSON object inside "
                    "<tool_call>...</tool_call> for each call."
                ),
                "The function.arguments value must be a JSON string. Do not invent tool names.",
            ]
        )
    else:
        sections.append(
            "Hermes auxiliary call. Answer directly and concisely; do not run tools, "
            "do not write files, and do not ask follow-up questions."
        )
    if model:
        sections.append(f"Hermes requested model hint: {model}")
    if tools:
        sections.append("Hermes-side tools:\n" + json.dumps(tools, ensure_ascii=False, sort_keys=True))
    if tool_choice is not None:
        sections.append("Tool choice:\n" + json.dumps(tool_choice, ensure_ascii=False, sort_keys=True))

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").upper()
        content = _render_content(message.get("content"))
        if message.get("tool_calls"):
            suffix = json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True)
            content = f"{content}\nTool calls: {suffix}".strip()
        if message.get("tool_call_id"):
            content = f"Tool call id: {message['tool_call_id']}\n{content}".strip()
        transcript.append(f"[{role}]\n{content}")
    sections.append("Conversation:\n" + "\n\n".join(transcript))
    return "\n\n".join(sections).strip() + "\n"


def extract_tool_calls(
    text: str,
    *,
    allowed_names: set[str] | frozenset[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Lift valid ``<tool_call>`` blocks into OpenAI tool call objects."""
    calls: list[dict[str, Any]] = []
    valid_spans: list[tuple[int, int]] = []
    for index, match in enumerate(_TOOL_CALL_RE.finditer(text)):
        try:
            payload = json.loads(match.group(1))
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        function = payload.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            normalized_function = function
        elif isinstance(payload.get("name"), str):
            normalized_function = {
                "name": payload["name"],
                "arguments": payload.get("arguments", "{}"),
            }
        else:
            continue
        if allowed_names is not None and normalized_function["name"] not in allowed_names:
            continue
        arguments = normalized_function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        calls.append(
            {
                "id": str(payload.get("id") or f"call_cursor_{index + 1}"),
                "type": "function",
                "function": {"name": normalized_function["name"], "arguments": arguments},
            }
        )
        valid_spans.append(match.span())

    if not valid_spans:
        return text.strip(), []
    pieces: list[str] = []
    cursor = 0
    for start, end in valid_spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    content = "".join(pieces).strip()
    return content, calls


def estimate_prompt_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    serialized = json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False)
    return max(1, (len(serialized) + 3) // 4)


class CursorStreamAccumulator:
    """Accumulate Cursor's line-delimited event stream."""

    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self.terminal = False
        self.is_error = False
        self.error_message = ""
        self.request_id = ""
        self.usage: dict[str, Any] = {}
        self._result_text = ""

    @property
    def text(self) -> str:
        return "".join(self._text_parts).strip() or self._result_text.strip()

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_parts).strip()

    def feed(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        event_type = event.get("type")
        if event_type == "thinking":
            text = event.get("text")
            if isinstance(text, str):
                self._reasoning_parts.append(text)
            return
        if event_type == "assistant":
            message = event.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                rendered = _render_content(content)
                if rendered:
                    self._text_parts.append(rendered)
            return
        if event_type != "result":
            return

        self.terminal = True
        self.is_error = bool(event.get("is_error")) or event.get("subtype") == "error"
        result = event.get("result")
        if isinstance(result, str):
            self._result_text = result
            if self.is_error:
                self.error_message = result
        request_id = event.get("request_id")
        if isinstance(request_id, str):
            self.request_id = request_id
        usage = event.get("usage")
        if isinstance(usage, dict):
            self.usage = usage

    def to_completion(
        self,
        *,
        model: str,
        prompt_tokens: int,
        allowed_tool_names: set[str] | frozenset[str] | None = None,
    ) -> dict[str, Any]:
        if self.is_error:
            raise RuntimeError(self.error_message or "Cursor CLI returned an error")
        if not self.terminal:
            raise RuntimeError("Cursor CLI stream ended without a terminal result event")

        content, tool_calls = extract_tool_calls(self.text, allowed_names=allowed_tool_names)
        output_tokens = self.usage.get("outputTokens")
        if not isinstance(output_tokens, int) or output_tokens < 0:
            output_tokens = max(1, (len(content) + 3) // 4) if content else 0
        cached_tokens = self.usage.get("cacheReadTokens")
        if not isinstance(cached_tokens, int) or cached_tokens < 0:
            cached_tokens = 0
        # Cursor's cache count includes its own agent harness and internal
        # rounds, while prompt_tokens estimates only the Hermes request. Keep
        # the OpenAI invariant that cached_tokens is a subset of prompt_tokens.
        cached_tokens = min(cached_tokens, prompt_tokens)
        finish_reason = "tool_calls" if tool_calls else "stop"
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": self.request_id or f"chatcmpl-cursor-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        }
