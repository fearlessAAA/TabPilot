"""Per-run in-memory state. The LLM never receives this object or its data frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

import pandas as pd

from tabpilot.tfm.context import ModelContext, ModelFeatures
from tabpilot.contracts import (
    PredictionArtifact,
    PredictionRow,
    PredictionType,
    SafeSchema,
)
from tabpilot.data.history_profile import NumericColumnProfile, build_numeric_column_profile


@dataclass(frozen=True)
class PreparedPredictionArtifact:
    prepared_id: str
    target_column: str
    prediction_type: PredictionType
    rows: list[PredictionRow]
    context: ModelContext
    type_evidence: str
    warnings: list[dict[str, object]]


@dataclass
class RunStore:
    run_id: str
    query: str
    history: pd.DataFrame
    schema: SafeSchema
    current: pd.DataFrame | None = None
    device: str = "cuda"
    max_context_rows: int = 100_000
    max_context_features: int | None = None
    random_state: int = 42
    tfm_backend: str = "tabpfn_v3_5"
    model_features: ModelFeatures | None = None
    _prepared: dict[str, PreparedPredictionArtifact] = field(default_factory=dict, init=False)
    _predictions: dict[str, PredictionArtifact] = field(default_factory=dict, init=False)
    _column_profile_cache: dict[str, NumericColumnProfile] = field(default_factory=dict, init=False, repr=False)
    _inspection_cache: dict[tuple[object, ...], dict[str, object]] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def put_prepared(
        self,
        *,
        target_column: str,
        prediction_type: PredictionType,
        rows: list[PredictionRow],
        context: ModelContext,
        type_evidence: str,
        warnings: list[dict[str, object]],
    ) -> PreparedPredictionArtifact:
        artifact = PreparedPredictionArtifact(
            prepared_id=f"prepared_{uuid4().hex}",
            target_column=target_column,
            prediction_type=prediction_type,
            rows=rows,
            context=context,
            type_evidence=type_evidence,
            warnings=warnings,
        )
        with self._lock:
            self._prepared[artifact.prepared_id] = artifact
        return artifact

    def get_prepared(self, prepared_id: str) -> PreparedPredictionArtifact | None:
        with self._lock:
            return self._prepared.get(prepared_id)

    def put_prediction(self, artifact: PredictionArtifact) -> None:
        with self._lock:
            self._predictions[artifact.prediction_id] = artifact

    def latest_prediction(self) -> PredictionArtifact | None:
        with self._lock:
            latest_id = next(reversed(self._predictions), None)
            return self._predictions.get(latest_id) if latest_id is not None else None

    def get_numeric_profile(self, column: str) -> NumericColumnProfile:
        with self._lock:
            profile = self._column_profile_cache.get(column)
            if profile is None:
                profile = build_numeric_column_profile(self.history[column])
                self._column_profile_cache[column] = profile
            return profile

    def get_cached_inspection(self, key: tuple[object, ...]) -> dict[str, object] | None:
        with self._lock:
            cached = self._inspection_cache.get(key)
            return dict(cached) if cached is not None else None

    def put_cached_inspection(self, key: tuple[object, ...], value: dict[str, object]) -> None:
        with self._lock:
            self._inspection_cache[key] = dict(value)

    @property
    def prediction_ids(self) -> list[str]:
        with self._lock:
            return list(self._predictions)
