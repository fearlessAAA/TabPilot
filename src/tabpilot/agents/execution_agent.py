"""Execution Agent: compiles a request into a structured prediction query through validated tools.

Predictive Reasoning path: the agent calls ``inspect_history`` (optional), ``prepare_tfm_input`` and
``tfm_predict`` one tool per turn until a prediction succeeds. Ranking and Filtering path: the agent
only submits the target and prediction type through ``submit_provided_table_intent``; the runtime
then predicts every candidate row itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from tabpilot.agents.progress import NO_PROGRESS_LIMIT, ProgressTracker
from tabpilot.agents.prompts import (
    MULTIPLE_TOOL_CALLS_FEEDBACK,
    PREMATURE_FINAL_ANSWER_FEEDBACK,
    PROVIDED_INTENT_REPAIR_FEEDBACK,
    REPEATED_INVALID_ACTION_FEEDBACK,
    REPEATED_SUCCESSFUL_ACTION_FEEDBACK,
    execution_messages,
    protocol_feedback,
    ranking_messages,
)
from tabpilot.agents.protocol import protocol_event, stable_seed, tool_messages
from tabpilot.contracts import ProtocolTraceEvent, RoleTraceEvent, TFMPredictArguments, ToolTraceEvent
from tabpilot.llm.client import SamplingProfile, ToolCallingClient, named_tool_choice
from tabpilot.tools.registry import ToolRegistry
from tabpilot.tools.tfm_predict import TFMPredictTool

RANKING_INTENT_ATTEMPTS = 2


@dataclass
class ExecutionResult:
    """Outcome of the Execution Agent; ``failure_code`` is ``None`` when a prediction exists."""

    events: list[ToolTraceEvent] = field(default_factory=list)
    protocol_events: list[ProtocolTraceEvent] = field(default_factory=list)
    role_events: list[RoleTraceEvent] = field(default_factory=list)
    failure_code: str | None = None


@dataclass
class ExecutionAgent:
    client: ToolCallingClient
    sampling: SamplingProfile
    max_turns: int
    seed: int

    def run_predictive_reasoning(
        self, query_id: str, query: str, registry: ToolRegistry, schema_view: dict[str, object]
    ) -> ExecutionResult:
        messages = execution_messages(query, schema_view)
        result = ExecutionResult()
        progress = ProgressTracker()
        for turn in range(1, self.max_turns + 1):
            seed = stable_seed(self.seed, query_id, turn)
            model_turn = self.client.complete(
                messages=messages,
                tools=registry.openai_definitions(),
                tool_choice="auto",
                seed=seed,
                sampling=self.sampling,
            )
            if not model_turn.tool_calls:
                result.protocol_events.append(protocol_event(turn, "tool_loop", "premature_final_answer", model_turn, seed))
                protocol_feedback(messages, PREMATURE_FINAL_ANSWER_FEEDBACK)
                continue
            if len(model_turn.tool_calls) != 1:
                result.protocol_events.append(protocol_event(turn, "tool_loop", "multiple_tool_calls", model_turn, seed))
                protocol_feedback(messages, MULTIPLE_TOOL_CALLS_FEEDBACK)
                continue

            call = model_turn.tool_calls[0]
            observation = registry.invoke(call.name, call.arguments)
            messages.extend(
                tool_messages(
                    call.call_id, call.name, call.arguments, model_turn.content, observation, full_observation=True
                )
            )
            result.events.append(
                ToolTraceEvent(
                    turn=turn,
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                    origin="model",
                    arguments=call.arguments,
                    observation=observation,
                    seed=seed,
                    generation=model_turn.generation,
                )
            )
            if observation.status == "error":
                result.failure_code = "tool_system_error"
                return result
            if call.name == "tfm_predict" and observation.status == "ok":
                return result
            progressed = progress.record(call.name, call.arguments, observation.status, observation.errors)
            if observation.status == "invalid" and not progressed:
                protocol_feedback(messages, REPEATED_INVALID_ACTION_FEEDBACK)
            elif progress.last_was_repeated_success:
                protocol_feedback(messages, REPEATED_SUCCESSFUL_ACTION_FEEDBACK)
            if progress.consecutive_no_progress >= NO_PROGRESS_LIMIT:
                result.failure_code = "no_progress"
                return result
        result.failure_code = "agent_iteration_limit"
        return result

    def run_ranking(
        self, query_id: str, query: str, registry: ToolRegistry, schema_view: dict[str, object]
    ) -> ExecutionResult:
        messages = ranking_messages(query, schema_view)
        result = ExecutionResult()
        for attempt in range(1, RANKING_INTENT_ATTEMPTS + 1):
            seed = stable_seed(self.seed, query_id, "provided_intent", attempt)
            model_turn = self.client.complete(
                messages=messages,
                tools=registry.openai_definitions(),
                tool_choice=named_tool_choice("submit_provided_table_intent"),
                seed=seed,
                sampling=self.sampling,
            )
            result.role_events.append(
                RoleTraceEvent(
                    role="ExecutionAgent",
                    turn=attempt,
                    code="intent_submitted" if model_turn.tool_calls else "missing_intent_tool_call",
                    seed=seed,
                    generation=model_turn.generation,
                )
            )
            if len(model_turn.tool_calls) != 1:
                code = "missing_intent_tool_call" if not model_turn.tool_calls else "multiple_intent_tool_calls"
                result.protocol_events.append(protocol_event(attempt, "tool_loop", code, model_turn, seed))
                protocol_feedback(messages, PROVIDED_INTENT_REPAIR_FEEDBACK)
                continue
            call = model_turn.tool_calls[0]
            observation = registry.invoke(call.name, call.arguments)
            result.events.append(
                ToolTraceEvent(
                    turn=attempt,
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                    origin="model",
                    arguments=call.arguments,
                    observation=observation,
                    seed=seed,
                    generation=model_turn.generation,
                )
            )
            if observation.status == "invalid":
                messages.extend(
                    tool_messages(
                        call.call_id, call.name, call.arguments, model_turn.content, observation, full_observation=False
                    )
                )
                protocol_feedback(messages, PROVIDED_INTENT_REPAIR_FEEDBACK)
                continue
            if observation.status == "error":
                result.failure_code = "provided_prepare_system_error"
                return result
            prepared_id = str((observation.data or {}).get("prepared_id", ""))
            if not prepared_id:
                result.failure_code = "provided_intent_invalid"
                return result
            prediction = TFMPredictTool().invoke(TFMPredictArguments(prepared_id=prepared_id), registry.store)
            result.events.append(
                ToolTraceEvent(
                    turn=attempt,
                    tool_call_id=f"runtime_predict_{attempt}",
                    tool_name="tfm_predict",
                    origin="runtime",
                    arguments=json.dumps({"prepared_id": prepared_id}, ensure_ascii=False, sort_keys=True),
                    observation=prediction,
                    seed=seed,
                    generation=None,
                )
            )
            if prediction.status != "ok":
                result.failure_code = "provided_predict_system_error"
            return result
        result.failure_code = "provided_intent_invalid"
        return result
