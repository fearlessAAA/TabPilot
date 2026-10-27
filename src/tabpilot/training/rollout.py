"""Multi-turn tool environment for on-policy Execution Agent training with ms-swift.

``ExecutionEnvironment`` is installed on ms-swift's ``MultiTurnScheduler``. It runs the real
``inspect_history`` and ``prepare_tfm_input`` tools on each rollout under the inference protocol
(one tool call per turn, the same protocol feedback, the no-progress limit, at most
``--max_turns`` turns, 4,096 tokens per turn and a 32,768-token context). A ``tfm_predict`` call on
a prepared query of the run ends the trajectory without running the TFM; an unknown
``prepared_id`` gets the tool's error and the loop continues. Subclasses turn the finished
trajectory into a training signal.

Also here: the colocated rollout loop with per-trajectory token caps, and a Qwen3.5 template
that renders protocol feedback after a tool observation as its own user turn.

Environment variables: ``TABPILOT_TASKS`` (``tasks.jsonl`` from ``scripts/build_training_data.py``),
``TABPILOT_MAX_TOKENS_PER_TURN``, ``TABPILOT_MAX_MODEL_LEN``, ``TABPILOT_PRESENCE_PENALTY`` and
optionally ``TABPILOT_ROLLOUT_LOG`` (one JSON line per finished trajectory).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tabpilot.agents.progress import NO_PROGRESS_LIMIT, ProgressTracker
from tabpilot.agents.prompts import (
    MULTIPLE_TOOL_CALLS_FEEDBACK,
    PREMATURE_FINAL_ANSWER_FEEDBACK,
    REPEATED_INVALID_ACTION_FEEDBACK,
    REPEATED_SUCCESSFUL_ACTION_FEEDBACK,
    protocol_feedback_text,
)
from tabpilot.agents.protocol import model_visible_observation_json
from tabpilot.contracts import TFMPredictArguments, ToolError, ToolResponse
from tabpilot.data.safe_schema import build_safe_schema
from tabpilot.data.topbench import load_task_bundle
from tabpilot.tools.inspect_history import InspectHistoryTool
from tabpilot.tools.prepare_tfm_input import PrepareTFMInputTool
from tabpilot.tools.registry import ToolRegistry
from tabpilot.tools.run_store import RunStore

TEMPLATE_TYPE = "qwen3_5_tabpilot"
NON_THINKING_PREFIX = "<think>\n\n</think>\n\n"
UNKNOWN_PREPARED_ID_MESSAGE = "prepared_id is not available in this run; call prepare_tfm_input first."
_TOOL_CALL = re.compile(r"<tool_call>\s*(?P<payload>.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION = re.compile(r"<function=(?P<name>[^>]+)>\s*(?P<parameters>.*?)\s*</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=(?P<name>[^>]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL)
# Tool parameters that the Qwen3.5 XML tool-call format carries as JSON text.
_JSON_PARAMETERS = {"prepare_tfm_input": {"rows"}, "inspect_history": {"columns", "max_examples"}}


@dataclass
class Trajectory:
    task_id: str
    task: dict[str, Any]
    schema: Any
    store: RunStore
    registry: ToolRegistry
    progress: ProgressTracker = field(default_factory=ProgressTracker)
    pending_messages: list[dict[str, str]] = field(default_factory=list)
    completion_tokens: int = 0
    generated_turns: int = 0
    last_prompt_tokens: int | None = None
    next_context_tokens: int | None = None
    # The latest prepare_tfm_input call: its target/type, and its canonical rows when valid.
    final_target: str | None = None
    final_type: str | None = None
    final_rows: list[dict[str, Any]] | None = None
    latest_prepared_id: str | None = None
    outstanding_repair: bool = False
    terminal: str | None = None
    stop_reason: str | None = None
    action_counts: dict[str, int] = field(default_factory=dict)


class ExecutionEnvironment:
    """Mixin body; ``register_scheduler`` combines it with ms-swift's ``MultiTurnScheduler``."""

    trajectory_class: type[Trajectory] = Trajectory

    def _setup(self) -> None:
        self._tasks = _read_jsonl_by_task(_required_env("TABPILOT_TASKS"))
        self._sources: dict[str, tuple[Any, Any]] = {}
        self._states: dict[int, Trajectory] = {}
        self._max_tokens_per_turn = int(os.environ.get("TABPILOT_MAX_TOKENS_PER_TURN", "4096"))
        self._max_model_len = int(os.environ.get("TABPILOT_MAX_MODEL_LEN", "32768"))

    # Subclasses produce the rollout_infos of a finished trajectory.
    def _finish(self, infer_request, state: Trajectory) -> dict[str, Any]:
        raise NotImplementedError

    async def on_trajectory_start(self, requests) -> None:
        for request in requests:
            task_id = str(request.data_dict.get("task_id", ""))
            if task_id not in self._tasks:
                raise KeyError(f"unknown training task_id: {task_id!r}")
            bundle, schema = self._source(task_id)
            store = RunStore(
                run_id=f"rollout_{request.uuid or id(request)}",
                query=bundle.query,
                history=bundle.history,
                schema=schema,
                current=None,
                device="cpu",
            )
            self._states[id(request)] = self.trajectory_class(
                task_id=task_id,
                task=self._tasks[task_id],
                schema=schema,
                store=store,
                registry=ToolRegistry([InspectHistoryTool(), PrepareTFMInputTool()], store),
            )

    async def on_turn_end(self, infer_request, response_choice, current_turn: int) -> dict[str, Any]:
        state = self._state(infer_request)
        token_ids = list(response_choice.token_ids or [])
        state.completion_tokens += len(token_ids)
        state.generated_turns += bool(token_ids)
        calls = _tool_calls(response_choice)
        done = False
        if len(calls) == 1:
            name, raw_arguments, payload = calls[0]
            state.action_counts[name or "malformed"] = state.action_counts.get(name or "malformed", 0) + 1
            done = self._act(state, name, raw_arguments, payload)
        elif not calls:
            state.pending_messages = [_feedback_message(PREMATURE_FINAL_ANSWER_FEEDBACK)]
        else:
            state.pending_messages = [_feedback_message(MULTIPLE_TOOL_CALLS_FEEDBACK)]
        if done:
            state.stop_reason = state.stop_reason or "terminal"
        elif self.max_turns and current_turn >= self.max_turns:
            done, state.stop_reason = True, "max_turns"
        elif state.progress.consecutive_no_progress >= NO_PROGRESS_LIMIT:
            done, state.stop_reason = True, "no_progress"
        if done:
            return {"done": True, "rollout_infos": self._complete(infer_request, state)}
        return {"done": False}

    def step(self, infer_request, response_choice, current_turn: int) -> dict[str, Any]:
        del current_turn
        state = self._state(infer_request)
        messages, state.pending_messages = state.pending_messages, []
        infer_request.messages.extend(messages)
        # Placeholder that the rollout loop removes once the next turn is generated.
        infer_request.messages.append({"role": "assistant", "content": None})
        model_ids = list(response_choice.token_ids or [])
        if state.last_prompt_tokens is not None:
            state.next_context_tokens = (
                state.last_prompt_tokens + len(model_ids) + len(self._observation_token_ids(messages, model_ids))
            )
        result: dict[str, Any] = {
            "infer_request": infer_request,
            "response_token_ids": model_ids,
            "response_loss_mask": [1] * len(model_ids),
        }
        logprobs = self._extract_logprobs_from_choice(response_choice)
        if logprobs:
            result["rollout_logprobs"] = logprobs
        return result

    def record_rollout_context(self, infer_request, output) -> None:
        prompt_tokens = getattr(getattr(output.response, "usage", None), "prompt_tokens", None)
        if prompt_tokens is not None:
            self._state(infer_request).last_prompt_tokens = int(prompt_tokens)

    def next_turn_max_tokens(self, infer_request, configured_max_tokens: int | None) -> int:
        state = self._state(infer_request)
        per_turn = min(configured_max_tokens or self._max_tokens_per_turn, self._max_tokens_per_turn)
        return max(min(per_turn, self._max_model_len - (state.next_context_tokens or 0)), 0)

    def terminate_for_length_limit(self, infer_request) -> dict[str, Any]:
        state = self._state(infer_request)
        state.stop_reason = "context_limit"
        return self._complete(infer_request, state)

    def check_finished(self, infer_request, response_choice, current_turn: int) -> bool:
        del infer_request, response_choice
        return bool(self.max_turns and current_turn >= self.max_turns)

    def _act(self, state: Trajectory, name: str, raw_arguments: str, payload: dict[str, Any] | None) -> bool:
        if name == "tfm_predict":
            return self._predict(state, raw_arguments, payload)
        observation = state.registry.invoke(name, raw_arguments)
        if name == "prepare_tfm_input":
            data = observation.data or {}
            if observation.status == "ok":
                rows = data.get("normalized_rows") or []
                state.final_rows = [dict(row.get("features") or {}) for row in rows]
                state.final_target = str(data.get("target_column", "")) or None
                state.final_type = str(data.get("requested_prediction_type", "")) or None
                state.latest_prepared_id = str(data.get("prepared_id", "")) or None
                state.outstanding_repair = False
            else:
                state.final_rows = None
                state.final_target = payload.get("target_column") if isinstance(payload, dict) else None
                state.final_type = payload.get("prediction_type") if isinstance(payload, dict) else None
                state.outstanding_repair = observation.status == "invalid" and bool(observation.errors)
        state.pending_messages = self._observation_messages(state, name, raw_arguments, observation)
        if observation.status == "error":
            state.stop_reason = "tool_error"
            return True
        return False

    def _predict(self, state: Trajectory, raw_arguments: str, payload: dict[str, Any] | None) -> bool:
        try:
            arguments = TFMPredictArguments.model_validate(payload)
        except ValidationError as error:
            observation = ToolResponse(
                status="invalid",
                errors=[ToolError(code="invalid_tool_arguments", path="prepared_id", message=str(error))],
            )
        else:
            if state.store.get_prepared(arguments.prepared_id) is not None:
                # At inference the TFM would run now and the Execution Agent would be done.
                latest = arguments.prepared_id == state.latest_prepared_id
                if latest and not state.outstanding_repair and state.final_rows is not None:
                    state.terminal = "predict_latest"
                else:
                    state.terminal = "predict_stale" if not latest else "predict_with_outstanding_repair"
                return True
            observation = ToolResponse(
                status="invalid",
                errors=[ToolError(code="unknown_prepared_id", path="prepared_id", message=UNKNOWN_PREPARED_ID_MESSAGE)],
            )
        state.pending_messages = self._observation_messages(state, "tfm_predict", raw_arguments, observation)
        return False

    @staticmethod
    def _observation_messages(
        state: Trajectory, name: str, raw_arguments: str, observation: ToolResponse
    ) -> list[dict[str, str]]:
        """The tool observation, then any protocol feedback as its own user message."""
        messages = [{"role": "tool", "content": model_visible_observation_json(observation, full_observation=True)}]
        progressed = state.progress.record(name, raw_arguments, observation.status, observation.errors)
        if observation.status == "invalid" and not progressed:
            messages.append(_feedback_message(REPEATED_INVALID_ACTION_FEEDBACK))
        elif state.progress.last_was_repeated_success:
            messages.append(_feedback_message(REPEATED_SUCCESSFUL_ACTION_FEEDBACK))
        return messages

    def _observation_token_ids(self, messages: list[dict[str, str]], model_ids: list[int]) -> list[int]:
        """Tokens the chat template inserts between this response and the next one (for the
        context-length forecast only; they are never trained on)."""
        rendered = "<|im_end|>\n"
        for message in messages:
            content = message["content"].strip()
            if message["role"] == "tool":
                content = f"<tool_response>\n{content}\n</tool_response>"
            rendered += f"<|im_start|>user\n{content}<|im_end|>\n"
        rendered += f"<|im_start|>assistant\n{NON_THINKING_PREFIX}"
        tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
        if model_ids and model_ids[-1] == tokenizer.convert_tokens_to_ids("<|im_end|>"):
            rendered = rendered.removeprefix("<|im_end|>")
        return list(tokenizer.encode(rendered, add_special_tokens=False))

    def _complete(self, infer_request, state: Trajectory) -> dict[str, Any]:
        info = {
            "task_id": state.task_id,
            "terminal": state.terminal,
            "stop_reason": state.stop_reason,
            "completion_tokens": state.completion_tokens,
            "action_counts": state.action_counts,
            **self._finish(infer_request, state),
        }
        log_path = os.environ.get("TABPILOT_ROLLOUT_LOG")
        if log_path:
            summary = {key: value for key, value in info.items() if isinstance(value, (str, int, float, dict, type(None)))}
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, default=str) + "\n")
        self._states.pop(id(infer_request), None)
        return info

    def _state(self, request) -> Trajectory:
        return self._states[id(request)]

    def _source(self, task_id: str) -> tuple[Any, Any]:
        if task_id not in self._sources:
            task = self._tasks[task_id]
            bundle = load_task_bundle(Path(task["task_dir"]), Path(task["query_path"]))
            schema = build_safe_schema(
                bundle.history, Path(task["task_dir"]), dropped_columns=bundle.dropped_columns, current=bundle.current
            )
            self._sources[task_id] = (bundle, schema)
        return self._sources[task_id]


def register_scheduler(name: str, environment: type[ExecutionEnvironment]) -> None:
    from swift.rollout import MultiTurnScheduler, multi_turns

    class Scheduler(environment, MultiTurnScheduler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._setup()

    multi_turns[name] = Scheduler


def _tool_calls(response_choice) -> list[tuple[str, str, dict[str, Any] | None]]:
    """``(name, canonical JSON arguments, parsed arguments)`` for each tool call in the turn."""
    parsed: list[tuple[str, Any]] = []
    for call in response_choice.message.tool_calls or []:
        try:
            parsed.append((str(call.function.name), json.loads(str(call.function.arguments))))
        except (TypeError, ValueError):
            parsed.append((str(call.function.name), str(call.function.arguments)))
    if not parsed:
        for block in _TOOL_CALL.findall(str(response_choice.message.content or "")):
            function = _FUNCTION.fullmatch(block.strip())
            if function is None:
                parsed.append(("", None))
                continue
            arguments = {item.group("name").strip(): item.group("value").strip() for item in _PARAMETER.finditer(function.group("parameters"))}
            parsed.append((function.group("name").strip(), arguments))
    calls = []
    for name, arguments in parsed:
        if not isinstance(arguments, dict):
            calls.append((name, arguments if isinstance(arguments, str) else "", None))
            continue
        payload = {key: _decode_json_parameter(name, key, value) for key, value in arguments.items()}
        calls.append((name, json.dumps(payload, ensure_ascii=False), payload))
    return calls


def _decode_json_parameter(tool: str, name: str, value: Any) -> Any:
    if name not in _JSON_PARAMETERS.get(tool, set()) or not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return value
    return value if isinstance(decoded, str) else decoded


def _feedback_message(text: str) -> dict[str, str]:
    return {"role": "user", "content": protocol_feedback_text(text)}


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _read_jsonl_by_task(path: str) -> dict[str, dict[str, Any]]:
    records = (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    return {str(record["task_id"]): record for record in records}


# ---------------------------------------------------------------------------------------------
# Rollout loop. ms-swift 4.5.2 shares one sampling config across all active trajectories; this
# fork of its colocated multi-turn loop batches later turns by each trajectory's remaining token
# cap, caps the first turn at the per-turn limit, and sets the presence penalty it does not expose.
# ---------------------------------------------------------------------------------------------


def install_rollout_loop() -> None:
    import swift.rlhf_trainers.rollout_mixin as rollout_mixin
    import swift.rollout.agent_loop as agent_loop

    agent_loop.run_multi_turn = run_multi_turn
    rollout_mixin.run_multi_turn = run_multi_turn
    trainer = rollout_mixin.RolloutTrainerMixin
    original_prepare = trainer._prepare_rollout_params

    def rollout(self, samples, request_config, is_global_inputs=False):
        if not getattr(request_config, "_tabpilot_bucket", False):
            cap = int(os.environ.get("TABPILOT_MAX_TOKENS_PER_TURN", "4096"))
            # ``_rollout`` re-reads the trainer's config, so resolve and cap it here instead.
            request_config = self._get_request_config()
            if request_config.max_tokens and request_config.max_tokens > cap:
                request_config = dataclasses.replace(request_config, max_tokens=cap)
        if self.vllm_mode == "server":
            return self._server_rollout(samples, request_config, is_global_inputs)
        return self._colocate_rollout(samples, request_config)

    def prepare_rollout_params(self) -> None:
        original_prepare(self)
        if os.environ.get("TABPILOT_PRESENCE_PENALTY") is not None:
            self.request_config.presence_penalty = float(os.environ["TABPILOT_PRESENCE_PENALTY"])

    trainer._rollout = rollout
    trainer._prepare_rollout_params = prepare_rollout_params


def run_multi_turn(requests, first_turn_outputs, scheduler, rollout_fn, request_config, max_turns=None, gather_fn=None):
    from swift.rollout.agent_loop import _identity_gather

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _run_multi_turn(requests, first_turn_outputs, scheduler, rollout_fn, request_config, max_turns, gather_fn or _identity_gather)
        )
    finally:
        loop.close()


async def _run_multi_turn(requests, outputs, scheduler, rollout_fn, request_config, max_turns, gather_fn):
    from swift.infer_engine.protocol import RolloutOutput
    from swift.rollout.agent_loop import extract_logprobs_from_choice, remove_response

    size = len(requests)
    finished: list[Any] = [None] * size
    infos: list[dict[str, Any]] = [{} for _ in range(size)]
    token_ids: list[list[list[int]]] = [[] for _ in range(size)]
    masks: list[list[list[int]]] = [[] for _ in range(size)]
    logprobs: list[list[list[float]]] = [[] for _ in range(size)]
    pending = list(range(size))
    turn = 1

    def add_chunk(
        index: int, ids: list[int], mask: list[int], chunk_logprobs: list[float], *, extend: bool, keep_empty: bool
    ) -> None:
        # A continued assistant message (``extend``) grows the last chunk instead of adding one.
        if extend and token_ids[index]:
            token_ids[index][-1].extend(ids)
            if masks[index]:
                masks[index][-1].extend(mask)
        elif ids or keep_empty:
            token_ids[index].append(ids)
            masks[index].append(mask)
        if chunk_logprobs:
            if extend and logprobs[index]:
                logprobs[index][-1].extend(chunk_logprobs)
            else:
                logprobs[index].append(chunk_logprobs)

    def finish(index: int, output: Any, *, append: bool, extend: bool) -> None:
        if append:
            choice = output.response.choices[0]
            ids = list(choice.token_ids or [])
            add_chunk(index, ids, [1] * len(ids), extract_logprobs_from_choice(choice), extend=extend, keep_empty=False)
        final_logprobs = logprobs[index]
        if sum(map(len, final_logprobs)) != sum(sum(mask) for mask in masks[index]):
            final_logprobs = []
        finished[index] = RolloutOutput(
            response=output.response,
            messages=requests[index].messages,
            response_token_ids=token_ids[index],
            response_loss_mask=masks[index],
            rollout_infos={**infos[index], "num_turns": turn},
            rollout_logprobs=final_logprobs,
        )

    while any(gather_fn([bool(pending)])):
        continuation = {}
        for index, output in zip(pending, outputs):
            messages = requests[index].messages
            if messages[-1]["content"] is None:
                remove_response(messages)
            completion = output.response.choices[0].message.content
            continuation[index] = messages[-1]["role"] == "assistant"
            if continuation[index]:
                messages[-1]["content"] += completion
            else:
                messages.append({"role": "assistant", "content": completion})
            scheduler.record_rollout_context(requests[index], output)
        results = await asyncio.gather(
            *[scheduler.on_turn_end(requests[i], o.response.choices[0], turn) for i, o in zip(pending, outputs)]
        )
        next_pending: list[int] = []
        for result, index, output in zip(results, pending, outputs):
            infos[index].update(result.get("rollout_infos") or {})
            choice = output.response.choices[0]
            if result.get("done") or (max_turns and turn >= max_turns):
                finish(index, output, append=True, extend=continuation[index])
                continue
            step = scheduler.step(requests[index], choice, turn)
            ids = list(step.get("response_token_ids") or [])
            mask = list(step.get("response_loss_mask") or [1] * len(ids))
            chunk_logprobs = step.get("rollout_logprobs") or extract_logprobs_from_choice(choice)
            add_chunk(index, ids, mask, chunk_logprobs, extend=continuation[index], keep_empty=True)
            requests[index] = step["infer_request"]
            if scheduler.next_turn_max_tokens(requests[index], request_config.max_tokens) <= 0:
                remove_response(requests[index].messages)
                infos[index].update(scheduler.terminate_for_length_limit(requests[index]))
                finish(index, output, append=False, extend=False)
                continue
            next_pending.append(index)
        turn += 1
        buckets: dict[int, list[int]] = {}
        for index in next_pending:
            buckets.setdefault(scheduler.next_turn_max_tokens(requests[index], request_config.max_tokens), []).append(index)
        produced: dict[int, Any] = {}
        for cap, indices in buckets.items():
            config = dataclasses.replace(request_config, max_tokens=cap)
            config._tabpilot_bucket = True
            produced.update(zip(indices, rollout_fn([requests[i] for i in indices], config)))
        pending = next_pending
        outputs = [produced[index] for index in pending]
    return finished


# ---------------------------------------------------------------------------------------------
# Template. Protocol feedback follows a tool observation as a separate user message
# (assistant -> tool -> user -> assistant), which ms-swift's ``swift`` backend cannot pair into
# query/response rounds. This Qwen3.5 template moves that user message into the tool's prompt,
# rendering exactly what the official chat template renders.
# ---------------------------------------------------------------------------------------------

_ASSISTANT_OPEN = "<|im_start|>assistant\n"
_TURN_CLOSE = "<|im_end|>\n"


def register_template() -> None:
    from swift.template import TEMPLATE_MAPPING
    from swift.template.register import register_template as register
    from swift.template.templates.qwen import Qwen3_5Template

    class TabPilotTemplate(Qwen3_5Template):
        def _swift_prepare_inputs(self, inputs):
            if self.template_backend != "swift":
                return super()._swift_prepare_inputs(inputs)
            feedback = _detach_feedback(inputs.messages)
            super()._swift_prepare_inputs(inputs)
            if any(item is not None for item in feedback):
                _attach_feedback(inputs.messages, feedback)
            return None

    if TEMPLATE_TYPE not in TEMPLATE_MAPPING:
        register(dataclasses.replace(TEMPLATE_MAPPING["qwen3_5"], template_type=TEMPLATE_TYPE, template_cls=TabPilotTemplate))


def _detach_feedback(messages: list[dict]) -> list[str | None]:
    """Remove the user message that follows each block of tool messages; return them per block."""
    feedback: list[str | None] = []
    index = 0
    while index < len(messages):
        if messages[index]["role"] != "tool":
            index += 1
            continue
        end = index
        while end + 1 < len(messages) and messages[end + 1]["role"] == "tool":
            end += 1
        follower = messages[end + 1] if end + 1 < len(messages) else None
        if follower is not None and follower["role"] == "user":
            feedback.append(str(follower["content"]))
            del messages[end + 1]
        else:
            feedback.append(None)
        index = end + 1
    return feedback


def _attach_feedback(messages: list[dict], feedback: list[str | None]) -> None:
    tool_messages = [message for message in messages if message["role"] == "tool"]
    if len(tool_messages) != len(feedback):
        raise ValueError(f"{len(tool_messages)} tool prompts for {len(feedback)} feedback slots")
    for message, text in zip(tool_messages, feedback):
        if text is None:
            continue
        content = message["content"]
        if not (isinstance(content, list) and content and isinstance(content[-1], str) and content[-1].endswith(_TURN_CLOSE + _ASSISTANT_OPEN)):
            raise ValueError("unexpected rendered tool prompt")
        content[-1] = f"{content[-1][: -len(_ASSISTANT_OPEN)]}<|im_start|>user\n{text}{_TURN_CLOSE}{_ASSISTANT_OPEN}"
