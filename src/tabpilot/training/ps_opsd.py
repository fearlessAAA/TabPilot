"""PS-OPSD with ms-swift's GKD trainer: on-policy distillation from a privileged teacher.

Registered scheduler: ``tabpilot_ps_opsd``. The student samples trajectories in the tool
environment of ``rollout.py``. For every generated turn ``t`` the student is scored on exactly
the prompt and response token IDs vLLM used, and the teacher (same backbone, no adapter) on the
same response after a prompt rendered from the same message history with the privileged
specification ``Q*`` (dataset column ``teacher_prompt``) inserted after the schema. The loss is
the forward KL with each vocabulary coordinate clipped,
``sum_v min(p_T(v) log(p_T(v) / p_S(v)), delta)``, averaged over the student's generated tokens,
with both distributions at the KD temperature.

``install()`` patches ms-swift 4.5.2's GKD trainer in memory: it gives ``GKDConfig`` the
multi-turn scheduler settings GRPO has, encodes samples turn by turn, and uses the clipped KL at
the KD temperature (rollouts keep their own sampling temperature). Settings:
``TABPILOT_KD_TEMPERATURE`` (1.1), ``TABPILOT_KL_CLIP`` (``delta``, 0.05) and
``TABPILOT_MAX_TURNS`` (10).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

from tabpilot.agents.prompts import PRIVILEGED_SPECIFICATION_OPEN_TAG, privileged_teacher_block
from tabpilot.training.rollout import ExecutionEnvironment, Trajectory

SCHEDULER = "tabpilot_ps_opsd"
SCHEMA_CLOSE_TAG = "</safe_schema>"


@dataclass
class TurnRecord:
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    messages_before_turn: list[dict[str, Any]]


@dataclass
class DistillationTrajectory(Trajectory):
    turns: list[TurnRecord] = field(default_factory=list)


class DistillationEnvironment(ExecutionEnvironment):
    trajectory_class = DistillationTrajectory

    def record_rollout_context(self, infer_request, output) -> None:
        """Capture the turn's prompt and response token IDs and the history before it."""
        response = output.response
        prompt_ids = list(getattr(response, "prompt_token_ids", None) or [])
        response_ids = list(response.choices[0].token_ids or []) if response.choices else []
        if response_ids and not prompt_ids:
            raise RuntimeError("rollout returned no prompt_token_ids; per-turn distillation needs them")
        if response_ids:
            # The history before this turn: drop the assistant message the loop just appended.
            history = copy.deepcopy(infer_request.messages[:-1])
            self._state(infer_request).turns.append(TurnRecord(prompt_ids, response_ids, history))
        super().record_rollout_context(infer_request, output)

    def _finish(self, infer_request, state: DistillationTrajectory) -> dict[str, Any]:
        del infer_request
        if len(state.turns) != state.generated_turns:
            raise RuntimeError(f"{state.task_id}: {len(state.turns)} recorded turns, {state.generated_turns} generated")
        return {"distillation_turns": list(state.turns)}


def teacher_messages(messages: list[dict[str, Any]], specification_json: str) -> list[dict[str, Any]]:
    """The student's messages with the privileged block inserted once, after the safe schema of
    the initial request; every later message is unchanged."""
    result = copy.deepcopy(messages)
    initial = [
        index
        for index, message in enumerate(result)
        if message.get("role") == "user" and "<user_query>" in str(message.get("content", ""))
    ]
    if len(initial) != 1 or str(result[initial[0]]["content"]).count(SCHEMA_CLOSE_TAG) != 1:
        raise ValueError("expected exactly one initial request message with one safe schema")
    if any(PRIVILEGED_SPECIFICATION_OPEN_TAG in str(message.get("content", "")) for message in result):
        raise ValueError("student messages already contain a privileged block")
    content = str(result[initial[0]]["content"])
    result[initial[0]]["content"] = content.replace(
        SCHEMA_CLOSE_TAG, f"{SCHEMA_CLOSE_TAG}\n\n{privileged_teacher_block(specification_json)}", 1
    )
    return result


def encode_turns(sample: Any, template: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One student row and one teacher row per generated turn; only response tokens are labeled."""
    turns: list[TurnRecord] = list((sample.rollout_infos or {}).get("distillation_turns") or [])
    if not turns or not sample.tools:
        raise RuntimeError("a distillation sample needs recorded turns and the tool schema")
    student_rows, teacher_rows = [], []
    for turn in turns:
        student_rows.append(_row(turn.prompt_token_ids, turn.response_token_ids))
        encoded = template.encode(
            {
                "messages": teacher_messages(turn.messages_before_turn, sample.teacher_prompt),
                "tools": copy.deepcopy(sample.tools),
                "chat_template_kwargs": {"enable_thinking": False},
                "add_eos": False,
            },
            return_length=True,
        )
        teacher_rows.append(_row(list(encoded["input_ids"]), turn.response_token_ids))
    return student_rows, teacher_rows


def _row(prompt_ids: list[int], response_ids: list[int]) -> dict[str, Any]:
    input_ids = list(prompt_ids) + list(response_ids)
    return {"input_ids": input_ids, "labels": [-100] * len(prompt_ids) + list(response_ids), "length": len(input_ids)}


def clipped_forward_kl(student_log_probs: Any, teacher_log_probs: Any, *, delta: float) -> Any:
    """Per position ``sum_v min(p_T(v) (log p_T(v) - log p_S(v)), delta)``."""
    import torch

    contribution = teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    return torch.clamp(contribution, max=delta).sum(-1)


def install() -> None:
    import swift.rlhf_trainers.arguments as arguments
    import swift.rlhf_trainers.gkd_trainer as gkd_trainer
    from swift.utils import to_device

    kd_temperature = float(os.environ.get("TABPILOT_KD_TEMPERATURE", "1.1"))
    delta = float(os.environ.get("TABPILOT_KL_CLIP", "0.05"))
    trainer = gkd_trainer.GKDTrainer
    original_post_init = arguments.GKDConfig.__post_init__

    def post_init(self) -> None:
        original_post_init(self)
        self.multi_turn_scheduler = SCHEDULER
        self.max_turns = int(os.environ.get("TABPILOT_MAX_TURNS", "10"))
        self.completion_length_limit_scope = "per_round"

    def encode_samples(self, samples):
        template = self.template
        student_rows: list[dict[str, Any]] = []
        teacher_rows: list[dict[str, Any]] = []
        with self._template_context(template):
            for sample in samples:
                student, teacher = encode_turns(sample, template)
                student_rows.extend(student)
                teacher_rows.extend(teacher)
            student_batch = to_device(template.data_collator(student_rows), self.accelerator.device)
            teacher_batch = to_device(template.data_collator(teacher_rows), self.accelerator.device)
        return student_batch, teacher_batch

    def compute_loss(self, student_logits, teacher_output, labels):
        import torch

        shifted_labels = torch.roll(labels, shifts=-1, dims=1)
        teacher_output.labels = torch.roll(teacher_output.labels, shifts=-1, dims=1)
        total, count = gkd_trainer.gkd_loss(
            student_logits,
            teacher_output,
            shifted_labels,
            self.beta,
            kd_temperature,
            kl_div_fn=lambda student, teacher: clipped_forward_kl(student, teacher, delta=delta),
        )
        return total * 0 if count == 0 else total / count

    arguments.GKDConfig.__post_init__ = post_init
    trainer._encode_samples = encode_samples
    trainer._compute_jsd_loss = compute_loss


def register() -> None:
    from tabpilot.training.rollout import register_scheduler

    register_scheduler(SCHEDULER, DistillationEnvironment)
