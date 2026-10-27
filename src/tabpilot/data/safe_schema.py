from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from tabpilot.contracts import ColumnKind, ColumnSpec, ProvidedCasesSpec, SafeSchema


MAX_INLINE_CATEGORY_VALUES = 32


def _safe_metadata(raw: object) -> dict[str, Any]:
    """Expose only target-neutral schema hints from benchmark metadata."""
    if not isinstance(raw, dict):
        return {}
    hints: dict[str, Any] = {}
    feature_types = raw.get("feature_types")
    if isinstance(feature_types, dict):
        hints["feature_types"] = {str(key): value for key, value in feature_types.items() if isinstance(value, str)}
    return hints


def _metadata(task_dir: Path) -> tuple[dict[str, Any], str]:
    for filename in ("info_mod.json", "info.json"):
        path = task_dir / filename
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                return _safe_metadata(json.load(handle)), filename
    return {}, "none"


def _kind(column: str, series: pd.Series, metadata: dict[str, Any]) -> ColumnKind:
    # pandas bool is technically numeric, but a bool feature is physically a
    # closed two-label representation. Check it first so `true` is not rejected
    # as a malformed numeric scalar later in the raw TabPFN contract.
    if is_bool_dtype(series.dtype):
        return ColumnKind.CATEGORICAL
    declared = metadata.get("feature_types", {}).get(column)
    if declared in {"numeric", "numerical", "number"}:
        return ColumnKind.NUMERIC
    if declared in {"categorical", "category"}:
        return ColumnKind.CATEGORICAL
    if is_numeric_dtype(series):
        return ColumnKind.NUMERIC
    if is_datetime64_any_dtype(series):
        return ColumnKind.DATETIME
    values = series.dropna().astype(str)
    # Long free text is reported as open text; the TFM receives it as a native text feature.
    if len(values) and (
        values.str.len().median() > 48 or values.nunique() > min(100, len(values) * 0.8)
    ):
        return ColumnKind.OPEN_TEXT
    return ColumnKind.CATEGORICAL


def _is_identifier(column: str) -> bool:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|\d+", column)
    }
    return bool(tokens & {"id", "index", "identifier"})


def _is_highly_unique(series: pd.Series) -> bool:
    non_null = series.dropna()
    return bool(len(non_null)) and non_null.nunique(dropna=True) / len(non_null) >= 0.95


def build_safe_schema(
    history: pd.DataFrame,
    task_dir: Path,
    *,
    dropped_columns: list[str] | None = None,
    current: pd.DataFrame | None = None,
) -> SafeSchema:
    metadata, source = _metadata(Path(task_dir))
    columns: list[ColumnSpec] = []
    for name in history.columns:
        kind = _kind(name, history[name], metadata)
        values: list[str | int | float | bool] = []
        if kind == ColumnKind.CATEGORICAL:
            # The cleaned history is the physical vocabulary TabPFN fits. Metadata
            # can identify a column as categorical, but it cannot extend that
            # vocabulary with values the fitted encoder has never observed.
            values = _history_category_values(history[name])
        columns.append(
            ColumnSpec(
                name=name,
                kind=kind,
                is_identifier=_is_identifier(name) and _is_highly_unique(history[name]),
                allowed_values=values,
            )
        )
    provided_cases = None
    if current is not None:
        current_columns = list(current.columns)
        current_column_set = set(current_columns)
        provided_cases = ProvidedCasesSpec(
            row_count=len(current),
            columns=current_columns,
            history_columns_absent=[name for name in history.columns if name not in current_column_set],
        )
    return SafeSchema(
        history_columns=columns,
        dropped_columns=dropped_columns or [],
        metadata_source=source,
        provided_cases=provided_cases,
    )


def project_safe_schema_for_model(schema: SafeSchema) -> dict[str, object]:
    """Return the bounded, oracle-free schema payload visible to the compiler LLM.

    Runtime retains every closed categorical domain in ``SafeSchema`` for physical
    validation.  The model sees a complete domain only when it has at most the
    fixed interface cap; larger domains are deferred rather than truncated.
    """
    columns: list[dict[str, object]] = []
    for column in schema.history_columns:
        payload: dict[str, object] = {
            "name": column.name,
            "kind": column.kind.value,
            "is_identifier": column.is_identifier,
        }
        if column.kind == ColumnKind.CATEGORICAL:
            domain_size = len(column.allowed_values)
            payload.update(
                {
                    "closed_domain": True,
                    "domain_size": domain_size,
                    "vocabulary_mode": (
                        "inline" if domain_size <= MAX_INLINE_CATEGORY_VALUES else "deferred"
                    ),
                    "allowed_values": (
                        column.allowed_values
                        if domain_size <= MAX_INLINE_CATEGORY_VALUES
                        else None
                    ),
                }
            )
        else:
            payload.update(
                {
                    "closed_domain": False,
                    "domain_size": None,
                    "vocabulary_mode": "not_applicable",
                    "allowed_values": None,
                }
            )
        columns.append(payload)
    payload: dict[str, object] = {
        "history_columns": columns,
        "dropped_columns": schema.dropped_columns,
        "max_inline_category_values": MAX_INLINE_CATEGORY_VALUES,
    }
    if schema.provided_cases is not None:
        payload["provided_cases"] = schema.provided_cases.model_dump(mode="json")
    return payload


_NUMERIC_CATEGORY_KINDS = frozenset(
    {"boolean", "decimal", "floating", "integer", "mixed-integer-float"}
)


def _history_category_values(series: pd.Series) -> list[str | int | float | bool]:
    """Return the closed, JSON-safe category vocabulary induced by history."""
    non_null = series.dropna()
    if non_null.empty:
        return []
    inferred = pd.api.types.infer_dtype(non_null, skipna=True)
    if inferred in _NUMERIC_CATEGORY_KINDS:
        raw_values = non_null.drop_duplicates().tolist()
    else:
        raw_values = non_null.map(str).drop_duplicates().tolist()
    return [
        value.item() if hasattr(value, "item") else value
        for value in raw_values
    ]
