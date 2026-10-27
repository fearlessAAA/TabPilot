"""OpenAI-compatible tool-calling client for the agents' LLM backbone (served by vLLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, Protocol, TypeAlias

from openai import BadRequestError, OpenAI

from tabpilot.contracts import GenerationTelemetry

NamedToolChoice: TypeAlias = dict[str, Any]
ToolChoice: TypeAlias = Literal["auto", "none", "required"] | NamedToolChoice

CONTEXT_OUTPUT_SAFETY_TOKENS = 128
MIN_CONTEXT_RETRY_OUTPUT_TOKENS = 256
_CONTEXT_LIMIT_PATTERN = re.compile(
    r"maximum context length is (?P<maximum>\d+) tokens.*?prompt contains at least "
    r"(?P<input>\d+) input tokens",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class SamplingProfile:
    """Decoding settings of one agent."""

    name: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_tokens: int


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelTurn:
    content: str
    tool_calls: list[ToolCall]
    reasoning_content: str = ""
    generation: GenerationTelemetry = field(default_factory=GenerationTelemetry)


class ToolCallingClient(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        seed: int,
        sampling: SamplingProfile,
        tool_choice: ToolChoice = "auto",
    ) -> ModelTurn: ...


def named_tool_choice(name: str) -> NamedToolChoice:
    return {"type": "function", "function": {"name": name}}


def tool_choice_mode(choice: ToolChoice) -> str:
    if isinstance(choice, dict):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        return f"named:{name}" if isinstance(name, str) else "named"
    return choice


@dataclass
class OpenAICompatibleToolClient:
    """Non-thinking chat completions against an OpenAI-compatible server."""

    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_seconds)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        seed: int,
        sampling: SamplingProfile,
        tool_choice: ToolChoice = "auto",
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "presence_penalty": sampling.presence_penalty,
            "seed": seed,
            "max_tokens": sampling.max_tokens,
            "extra_body": {
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "repetition_penalty": sampling.repetition_penalty,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        if tools is not None:
            request["tools"] = tools
            request["tool_choice"] = tool_choice
        started = perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except BadRequestError as error:
            # Shrink the output budget only when the server reports a context overflow.
            retry_tokens = _context_budget_retry_tokens(sampling, str(error))
            if retry_tokens is None:
                raise
            request["max_tokens"] = retry_tokens
            response = self._client.chat.completions.create(**request)
        latency_seconds = perf_counter() - started
        message = response.choices[0].message
        reasoning_content = _reasoning_text(message)
        visible_content = _visible_text(message)
        usage = getattr(response, "usage", None)
        return ModelTurn(
            content=visible_content,
            tool_calls=[
                ToolCall(call_id=call.id, name=call.function.name, arguments=call.function.arguments)
                for call in (message.tool_calls or [])
            ],
            reasoning_content=reasoning_content,
            generation=GenerationTelemetry(
                finish_reason=response.choices[0].finish_reason,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
                reasoning_content_present=bool(reasoning_content),
                reasoning_content_length=len(reasoning_content),
                visible_content_length=len(visible_content),
                latency_seconds=latency_seconds,
                tool_choice_mode=tool_choice_mode(tool_choice),
            ),
        )


def _context_budget_retry_tokens(profile: SamplingProfile, error_text: str) -> int | None:
    match = _CONTEXT_LIMIT_PATTERN.search(error_text)
    if match is None:
        return None
    permitted = int(match.group("maximum")) - int(match.group("input")) - CONTEXT_OUTPUT_SAFETY_TOKENS
    if permitted < MIN_CONTEXT_RETRY_OUTPUT_TOKENS or permitted >= profile.max_tokens:
        return None
    return permitted


def _reasoning_text(message: object) -> str:
    """Read the reasoning channel (never persisted, only measured)."""
    candidates: list[object] = [getattr(message, "reasoning", None), getattr(message, "reasoning_content", None)]
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        candidates.extend([model_extra.get("reasoning"), model_extra.get("reasoning_content")])
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def _visible_text(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text") if isinstance(item, dict) else getattr(item, "text", None) for item in content]
        return "".join(part for part in parts if isinstance(part, str))
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict) and isinstance(model_extra.get("content"), str):
        return model_extra["content"]
    return ""
