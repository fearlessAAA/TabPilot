"""Loading of ``configs/tabpilot.yaml``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from tabpilot.llm.client import SamplingProfile
from tabpilot.tfm.context import ModelFeatures
from tabpilot.tfm.predictor import DEFAULT_TFM_BACKEND, TFM_BACKENDS

_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "max_tokens")
_MODEL_FEATURE_KINDS = ("datetime", "text", "identifiers")


@dataclass(frozen=True)
class TabPilotConfig:
    seed: int
    max_turns: int
    execution_sampling: SamplingProfile
    response_sampling: SamplingProfile
    tfm_backend: str
    max_context_rows: int
    max_context_features: int | None
    model_features: ModelFeatures

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "max_turns": self.max_turns,
            "execution_sampling": sampling_dict(self.execution_sampling),
            "response_sampling": sampling_dict(self.response_sampling),
            "tfm_backend": self.tfm_backend,
            "max_context_rows": self.max_context_rows,
            "max_context_features": self.max_context_features,
            "model_features": asdict(self.model_features),
        }


def load_config(path: Path) -> TabPilotConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    execution = payload["execution_agent"]
    tfm = payload["tfm"]
    backend = tfm.get("backend", DEFAULT_TFM_BACKEND)
    if backend not in TFM_BACKENDS:
        raise ValueError(f"tfm.backend must be one of {sorted(TFM_BACKENDS)}, got {backend!r}")
    profile = TFM_BACKENDS[backend]
    max_rows = int(tfm["max_context_rows"])
    max_features = tfm.get("max_context_features")
    if not 0 < max_rows <= profile.max_context_rows:
        raise ValueError(f"tfm.max_context_rows must be in (0, {profile.max_context_rows}] for {backend}")
    if max_features is not None and not 0 < int(max_features) <= profile.max_context_features:
        raise ValueError(f"tfm.max_context_features must be null or in (0, {profile.max_context_features}] for {backend}")
    return TabPilotConfig(
        seed=int(payload["seed"]),
        max_turns=int(execution["max_turns"]),
        execution_sampling=_sampling("execution", execution["sampling"]),
        response_sampling=_sampling("response", payload["response_agent"]["sampling"]),
        tfm_backend=backend,
        max_context_rows=max_rows,
        max_context_features=None if max_features is None else int(max_features),
        model_features=_model_features(tfm.get("model_features"), backend),
    )


def sampling_dict(profile: SamplingProfile) -> dict[str, object]:
    return {"name": profile.name, **{name: getattr(profile, name) for name in _SAMPLING_FIELDS}}


def _sampling(name: str, raw: dict[str, Any]) -> SamplingProfile:
    missing = [field for field in _SAMPLING_FIELDS if field not in raw]
    if missing or set(raw) - set(_SAMPLING_FIELDS):
        raise ValueError(f"{name} sampling needs exactly the keys {_SAMPLING_FIELDS}")
    return SamplingProfile(name=name, **raw)


def _model_features(raw: dict[str, Any] | None, backend: str) -> ModelFeatures:
    """``tfm.model_features`` (default: every kind the backend officially supports) may only narrow it."""
    profile = TFM_BACKENDS[backend]
    supported = ModelFeatures(datetime=profile.supports_datetime, text=profile.supports_text, identifiers=True)
    if raw is None:
        return supported
    if set(raw) != set(_MODEL_FEATURE_KINDS) or not all(isinstance(value, bool) for value in raw.values()):
        raise ValueError(f"tfm.model_features must map exactly {_MODEL_FEATURE_KINDS} to booleans")
    requested = ModelFeatures(**raw)
    for kind in _MODEL_FEATURE_KINDS:
        if getattr(requested, kind) and not getattr(supported, kind):
            raise ValueError(f"tfm.model_features.{kind} is not supported by {backend}")
    return requested
