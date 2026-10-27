"""SA-GRPO with ms-swift: rollout scheduler, Structure-Aware Reward and group resampling.

Registered names (``ms_swift_plugin.py``): scheduler ``tabpilot_sa_grpo``, reward
``tabpilot_structure_aware_reward``, callback ``tabpilot_skip_no_signal_steps``.

* The scheduler scores each finished trajectory: ``V = 1`` when it ends with ``tfm_predict`` on the
  latest valid prepared query with no unrepaired invalid call; the target, type and canonical rows
  of that query are compared with the reference query from ``TABPILOT_TASKS``. The TFM never runs.
* Groups whose ``G`` rewards are all equal carry no advantage. They are resampled up to
  ``--max_resample_times`` times (this replaces ms-swift 4.5.2's ``_dynamic_sampling``, whose
  last retry is generated but never checked); a step whose groups all stay flat skips the
  optimizer and LR-scheduler update.
"""

from __future__ import annotations

import math
from typing import Any

from tabpilot.training.reference_query import ReferenceQuery
from tabpilot.training.rollout import ExecutionEnvironment, Trajectory
from tabpilot.training.structure_aware_reward import column_types, structure_aware_reward

SCHEDULER = "tabpilot_sa_grpo"
REWARD = "tabpilot_structure_aware_reward"
CALLBACK = "tabpilot_skip_no_signal_steps"
_SIGNAL_THRESHOLD = 1e-8


class StructureAwareRewardEnvironment(ExecutionEnvironment):
    def _setup(self) -> None:
        super()._setup()
        self._columns: dict[str, Any] = {}
        self._references = {
            task_id: ReferenceQuery.from_json(task["reference_query"])
            for task_id, task in self._tasks.items()
            if task.get("reference_query") is not None
        }

    def _finish(self, infer_request, state: Trajectory) -> dict[str, Any]:
        del infer_request
        reference = self._references[state.task_id]
        if state.task_id not in self._columns:
            bundle, schema = self._source(state.task_id)
            self._columns[state.task_id] = column_types(schema, bundle.history)
        breakdown = structure_aware_reward(
            valid=state.terminal == "predict_latest",
            target_column=state.final_target,
            prediction_type=state.final_type,
            rows=state.final_rows,
            reference_target=reference.target_column,
            reference_type=reference.prediction_type,
            reference_rows=list(reference.rows.values()),
            schema_names=list(state.schema.names),
            columns=self._columns[state.task_id],
        )
        return breakdown.to_dict()


class StructureAwareReward:
    """ORM: returns the reward the scheduler computed for each trajectory."""

    def __init__(self, args: object = None, **kwargs: object) -> None:
        del args, kwargs

    def __call__(self, completions: list[str], rollout_infos: list[dict[str, Any]] | None = None, **kwargs: Any) -> list[float]:
        del kwargs
        if rollout_infos is None or len(rollout_infos) != len(completions):
            raise ValueError("every completion needs the scheduler's rollout_infos")
        rewards = [float(info["R"]) for info in rollout_infos]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in rewards):
            raise ValueError(f"reward out of [0, 1]: {rewards}")
        return rewards


def group_has_signal(rewards: list[float], group_size: int) -> list[bool]:
    if group_size <= 0 or len(rewards) % group_size:
        raise ValueError(f"{len(rewards)} rewards do not form groups of {group_size}")
    if not all(math.isfinite(value) for value in rewards):
        raise ValueError(f"non-finite reward: {rewards}")
    groups = [rewards[start:start + group_size] for start in range(0, len(rewards), group_size)]
    return [max(group) - min(group) > _SIGNAL_THRESHOLD for group in groups]


def _combined_rewards(trainer: Any, rewards_per_func) -> list[float]:
    weights = getattr(trainer, "reward_weights", None)
    combined = (rewards_per_func * weights.unsqueeze(0)).sum(dim=1) if weights is not None else rewards_per_func.sum(dim=1)
    return [float(value) for value in combined.tolist()]


def dynamic_sampling(self, samples, rewards_per_func):
    """Keep whole groups with reward variance, generating up to ``max_resample_times`` new
    batches; if too few are found, fall back to the original batch."""
    import torch
    from accelerate.utils import gather_object

    group_size = int(self.num_generations)
    target = int(self.args.generation_batch_size)
    original = (samples, rewards_per_func)
    kept_samples: list[Any] = []
    kept_rewards: list[Any] = []
    attempt = 0
    while True:
        all_samples = gather_object(samples)
        flags = group_has_signal(_combined_rewards(self, rewards_per_func), group_size)
        for index, has_signal in enumerate(flags):
            if has_signal and len(kept_samples) < target:
                kept_samples.extend(all_samples[index * group_size:(index + 1) * group_size])
                kept_rewards.append(rewards_per_func[index * group_size:(index + 1) * group_size])
        if len(kept_samples) >= target or attempt >= int(self.max_resample_times):
            break
        inputs = next(self.dynamic_resample_iterator)
        if self.template.truncation_strategy == "raise":
            inputs = self.resample_encode_failed_inputs(inputs)
        samples = self._generate_completions(self.to_samples(inputs))
        rewards_per_func = self._compute_rewards_per_func(samples)
        attempt += 1
    if len(kept_samples) < target:
        return original
    local = slice(self.accelerator.process_index * len(samples), (self.accelerator.process_index + 1) * len(samples))
    return kept_samples[:target][local], torch.cat(kept_rewards)[:target]


class SkipNoSignalSteps:
    """Turns the optimizer and LR-scheduler steps into no-ops when no group has reward variance."""

    def __init__(self, args: object, trainer: Any) -> None:
        del args
        self._trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        del args, state, kwargs
        rewards = _combined_rewards(self._trainer, self._trainer._rewards_per_func)
        if not any(group_has_signal(rewards, int(self._trainer.num_generations))):
            _skip_once(self._trainer.optimizer, clear_grads=self._trainer.model)
            _skip_once(self._trainer.lr_scheduler)
        return control


def _skip_once(target: Any, clear_grads: Any = None) -> None:
    original = target.step

    def skipped(*args: object, **kwargs: object) -> None:
        del args, kwargs
        target.step = original
        if clear_grads is not None:
            for parameter in clear_grads.parameters():
                parameter.grad = None

    target.step = skipped


def register() -> None:
    from swift.callbacks import TrainerCallback, callbacks_map
    from swift.rewards import ORM, orms
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

    from tabpilot.training.rollout import register_scheduler

    register_scheduler(SCHEDULER, StructureAwareRewardEnvironment)
    orms[REWARD] = type("StructureAwareRewardORM", (StructureAwareReward, ORM), {})
    callbacks_map[CALLBACK] = type("SkipNoSignalStepsCallback", (SkipNoSignalSteps, TrainerCallback), {})
    GRPOTrainer._dynamic_sampling = dynamic_sampling
