"""Typed contracts shared by the agents, the tools and the run traces."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tabpilot.agents.prompts import (
    FEATURES_FIELD_DESCRIPTION,
    INSPECT_COLUMNS_FIELD_DESCRIPTION,
    INSPECT_MAX_EXAMPLES_FIELD_DESCRIPTION,
    INSPECT_VIEW_FIELD_DESCRIPTION,
    PREDICTION_TYPE_FIELD_DESCRIPTION,
    PREPARED_ID_FIELD_DESCRIPTION,
    PROVIDED_PREDICTION_TYPE_FIELD_DESCRIPTION,
    PROVIDED_TARGET_COLUMN_FIELD_DESCRIPTION,
    ROW_ID_FIELD_DESCRIPTION,
    ROWS_FIELD_DESCRIPTION,
    TARGET_COLUMN_FIELD_DESCRIPTION,
)

Scalar = str | int | float | bool
PredictionType = Literal["classification", "regression"]


class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    OPEN_TEXT = "open_text"


class ColumnSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: ColumnKind
    is_identifier: bool = False
    allowed_values: list[Scalar] = Field(default_factory=list)


class ProvidedCasesSpec(BaseModel):
    """Structure-only view of runtime-bound candidate cases."""

    model_config = ConfigDict(extra="forbid")

    row_count: int = Field(ge=1)
    columns: list[str]
    history_columns_absent: list[str]


class SafeSchema(BaseModel):
    """Oracle-free schema metadata exposed to model roles."""

    model_config = ConfigDict(extra="forbid")

    history_columns: list[ColumnSpec]
    dropped_columns: list[str] = Field(default_factory=list)
    metadata_source: Literal["info_mod.json", "info.json", "none"]
    provided_cases: ProvidedCasesSpec | None = None

    @property
    def names(self) -> list[str]:
        return [column.name for column in self.history_columns]

    def by_name(self, name: str) -> ColumnSpec:
        for column in self.history_columns:
            if column.name == name:
                return column
        raise KeyError(name)


class PredictionRow(BaseModel):
    """A sparse model-authored prediction case."""

    model_config = ConfigDict(extra="forbid")

    row_id: str = Field(min_length=1, max_length=128, description=ROW_ID_FIELD_DESCRIPTION)
    features: dict[str, Scalar | None] = Field(default_factory=dict, description=FEATURES_FIELD_DESCRIPTION)


class PrepareTFMInputArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_column: str = Field(min_length=1, description=TARGET_COLUMN_FIELD_DESCRIPTION)
    prediction_type: PredictionType = Field(description=PREDICTION_TYPE_FIELD_DESCRIPTION)
    rows: list[PredictionRow] = Field(
        default_factory=list,
        max_length=128,
        description=ROWS_FIELD_DESCRIPTION,
    )


class ProvidedTableIntentArguments(BaseModel):
    """The only model-authored decision exposed in provided-table mode."""

    model_config = ConfigDict(extra="forbid")

    target_column: str = Field(min_length=1, description=PROVIDED_TARGET_COLUMN_FIELD_DESCRIPTION)
    prediction_type: PredictionType = Field(description=PROVIDED_PREDICTION_TYPE_FIELD_DESCRIPTION)


class InspectHistoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    view: Literal["profiles", "rows"] = Field(description=INSPECT_VIEW_FIELD_DESCRIPTION)
    columns: list[str] | None = Field(default=None, description=INSPECT_COLUMNS_FIELD_DESCRIPTION)
    max_examples: int = Field(default=2, ge=0, le=3, description=INSPECT_MAX_EXAMPLES_FIELD_DESCRIPTION)


class TFMPredictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prepared_id: str = Field(min_length=1, description=PREPARED_ID_FIELD_DESCRIPTION)


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    path: str | None = None
    message: str
    allowed_values: list[Scalar] | None = None
    submitted_value: Scalar | None = None
    candidate_values: list[Scalar] | None = None
    retryable: bool = True


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "invalid", "error"]
    data: dict[str, Any] | None = None
    errors: list[ToolError] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: str
    predicted_label: Scalar | None = None
    predicted_value: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class PredictionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    prepared_id: str
    target_column: str
    prediction_type: PredictionType
    records: list[PredictionRecord]
    model_version: str
    elapsed_seconds: float
    history_rows_used: int
    feature_count: int
    sampling_strategy: str
    sampling_seed: int
    prediction_backend: Literal[
        "tabpfn_v3_5_direct",
        "tabpfn_v3_5_many_class",
        "tabicl_v2_direct",
        "tabicl_v2_many_class",
        "limix_2_direct",
    ] = "tabpfn_v3_5_direct"
    target_class_count: int | None = None
    native_class_limit: int | None = None
    probability_semantics: Literal[
        "native_tabpfn", "native_tfm", "many_class_decoded", "not_applicable"
    ] = "not_applicable"
    target_label_encoding: Literal["contiguous_integer_ids", "not_applicable"] = "not_applicable"
    many_class_config: dict[str, Any] = Field(default_factory=dict)
    backend_config: dict[str, Any] = Field(default_factory=dict)


class RawTaskBundle(BaseModel):
    """Runtime data with no ground-truth fields by construction."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    query_id: str
    query: str
    query_path: str
    task_dir: str
    history: Any
    current: Any | None = None
    history_source_id: str = ""
    dropped_columns: list[str] = Field(default_factory=list)


class ToolTraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int
    tool_call_id: str
    tool_name: str
    origin: Literal["model", "runtime"] = "model"
    arguments: str
    observation: ToolResponse
    seed: int
    # Every model turn that emitted a tool call is part of the experimental
    # cost.  Keeping its telemetry beside the deterministic observation makes
    # post-run efficiency analysis complete without exposing hidden reasoning.
    generation: GenerationTelemetry | None = None


class GenerationTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_content_present: bool = False
    reasoning_content_length: int = 0
    visible_content_length: int = 0
    latency_seconds: float | None = None
    tool_choice_mode: str | None = None


class ProtocolTraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int
    phase: Literal["tool_loop", "finalization"]
    code: str
    content: str | None = None
    seed: int
    generation: GenerationTelemetry | None = None


class RoleTraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["ExecutionAgent", "ResponseAgent"]
    turn: int
    code: str
    seed: int
    generation: GenerationTelemetry | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RunTrace(BaseModel):
    """Everything one TabPilot run records; written as ``trace.json``."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    query_id: str
    status: Literal["success", "failure"]
    failure_code: str | None = None
    answer: str | None = None
    materialized_table_path: str | None = None
    events: list[ToolTraceEvent] = Field(default_factory=list)
    protocol_events: list[ProtocolTraceEvent] = Field(default_factory=list)
    role_events: list[RoleTraceEvent] = Field(default_factory=list)
    prediction_ids: list[str] = Field(default_factory=list)
    case_key: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
