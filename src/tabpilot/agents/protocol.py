"""Message and trace conventions shared by the agents and by training rollouts."""

from __future__ import annotations

import hashlib
import json
import re

from tabpilot.contracts import ProtocolTraceEvent, ToolResponse


def stable_seed(global_seed: int, query_id: str, *parts: object) -> int:
    """Per-call sampling seed derived from the run seed, the request and the call position."""
    suffix = ":".join(map(str, parts))
    return int(hashlib.sha256(f"{global_seed}:{query_id}:{suffix}".encode()).hexdigest()[:8], 16)


def sanitize_visible_text(content: str) -> str:
    """Drop any ``<think>`` block; reasoning text never enters messages, traces or answers."""
    return re.sub(r"<think>.*?</think>", "", content or "", flags=re.IGNORECASE | re.DOTALL).strip()


def tool_messages(
    call_id: str,
    name: str,
    arguments: str,
    content: str,
    observation: ToolResponse,
    *,
    full_observation: bool,
) -> list[dict[str, object]]:
    """The assistant tool call and its tool observation, as appended to the conversation."""
    return [
        {
            "role": "assistant",
            "content": sanitize_visible_text(content) or None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": model_visible_observation_json(observation, full_observation=full_observation),
        },
    ]


def model_visible_observation_json(observation: ToolResponse, *, full_observation: bool = False) -> str:
    """Serialize a tool observation for the next model turn.

    The Predictive Reasoning path shows the complete observation. The Ranking and Filtering path
    drops trace-only provenance and the per-row list of unstated (natively missing) features,
    which is not an actionable repair.
    """
    if full_observation:
        return observation.model_dump_json()
    payload = observation.model_dump(mode="json")
    payload["provenance"] = {}
    data = payload.get("data")
    if isinstance(data, dict):
        warnings = data.get("warnings")
        if isinstance(warnings, list):
            data["warnings"] = [
                warning
                for warning in warnings
                if not (isinstance(warning, dict) and warning.get("code") == "unstated_features_missing")
            ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def protocol_event(turn: int, phase: str, code: str, model_turn: object, seed: int) -> ProtocolTraceEvent:
    content = str(getattr(model_turn, "content", ""))
    return ProtocolTraceEvent(
        turn=turn,
        phase=phase,  # type: ignore[arg-type]
        code=code,
        content=sanitize_visible_text(content) or None,
        seed=seed,
        generation=getattr(model_turn, "generation", None),
    )
