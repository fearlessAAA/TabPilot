"""Response Agent: a tool-free LLM that turns the locked TFM predictions into the final answer."""

from __future__ import annotations

from dataclasses import dataclass

from tabpilot.agents.prompts import (
    EMPTY_RESPONSE_FEEDBACK,
    RESPONSE_TOOL_CALL_FEEDBACK,
    protocol_feedback,
    response_messages,
)
from tabpilot.agents.protocol import protocol_event, sanitize_visible_text, stable_seed
from tabpilot.contracts import PredictionRow, ProtocolTraceEvent, RoleTraceEvent
from tabpilot.llm.client import SamplingProfile, ToolCallingClient

RESPONSE_ATTEMPTS = 2


@dataclass
class ResponseAgent:
    client: ToolCallingClient
    sampling: SamplingProfile
    seed: int
    # Response seeds continue after the Execution Agent's turn budget.
    turn_offset: int

    def respond(
        self, query_id: str, query: str, facts: list[dict[str, object]]
    ) -> tuple[str | None, list[ProtocolTraceEvent], list[RoleTraceEvent]]:
        messages = response_messages(query=query, case_prediction_facts=facts)
        protocol_events: list[ProtocolTraceEvent] = []
        for attempt in range(1, RESPONSE_ATTEMPTS + 1):
            turn = self.turn_offset + attempt
            seed = stable_seed(self.seed, query_id, turn)
            model_turn = self.client.complete(
                messages=messages, tools=None, tool_choice="none", seed=seed, sampling=self.sampling
            )
            answer = sanitize_visible_text(model_turn.content)
            if answer:
                protocol_events.append(protocol_event(turn, "finalization", "final_answer", model_turn, seed))
                role = RoleTraceEvent(
                    role="ResponseAgent", turn=attempt, code="final_answer", seed=seed, generation=model_turn.generation
                )
                return answer, protocol_events, [role]
            code = "finalization_tool_call" if model_turn.tool_calls else "empty_final_answer"
            protocol_events.append(protocol_event(turn, "finalization", code, model_turn, seed))
            protocol_feedback(messages, RESPONSE_TOOL_CALL_FEEDBACK if model_turn.tool_calls else EMPTY_RESPONSE_FEEDBACK)
        return None, protocol_events, []


def prediction_facts(rows: list[PredictionRow], prediction: dict[str, object]) -> list[dict[str, object]]:
    """Join each submitted row with its locked prediction, in submission order."""
    records = prediction.get("records")
    if not isinstance(records, list):
        raise RuntimeError("prediction artifact has no records")
    by_row_id = {
        record.get("row_id"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("row_id"), str)
    }
    facts: list[dict[str, object]] = []
    for row in rows:
        record = by_row_id.get(row.row_id)
        if record is None:
            raise RuntimeError(f"prediction artifact lacks row_id {row.row_id!r}")
        facts.append(
            {
                "row_id": row.row_id,
                "target_column": prediction.get("target_column"),
                "prediction_type": prediction.get("prediction_type"),
                "stated_features": row.features,
                "predicted_label": record.get("predicted_label"),
                "predicted_value": record.get("predicted_value"),
                "probabilities": record.get("probabilities", {}),
            }
        )
    return facts
