"""Bounded, target-neutral inspection of cleaned history for the ReAct agent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabpilot.agents.prompts import (
    INSPECTION_POPULATION_NOTE,
    INSPECT_HISTORY_TOOL_DESCRIPTION,
)
from tabpilot.contracts import ColumnKind, InspectHistoryArguments, ToolError, ToolResponse
from tabpilot.tools.run_store import RunStore


MAX_ENUMERATED_NUMERIC_VALUES = 20


class InspectHistoryTool:
    name = "inspect_history"
    description = INSPECT_HISTORY_TOOL_DESCRIPTION
    input_model = InspectHistoryArguments

    def invoke(self, arguments: InspectHistoryArguments, store: RunStore) -> ToolResponse:
        resolved_columns, errors = _resolve_columns(arguments, store)
        if errors:
            return ToolResponse(status="invalid", errors=errors)
        key = (arguments.view, tuple(resolved_columns), arguments.max_examples)
        cached = store.get_cached_inspection(key)
        if cached is not None:
            return ToolResponse(
                status="ok",
                data=cached,
                provenance=_provenance(arguments, resolved_columns, store),
            )

        if arguments.view == "profiles":
            data: dict[str, object] = {
                "note": INSPECTION_POPULATION_NOTE,
                "view": "profiles",
                "profiles": [
                    _profile_column(name, max_examples=arguments.max_examples, store=store)
                    for name in resolved_columns
                ],
            }
        else:
            data = {
                "note": INSPECTION_POPULATION_NOTE,
                "view": "rows",
                "columns": resolved_columns,
                "example_rows": _example_rows(
                    columns=resolved_columns,
                    max_examples=arguments.max_examples,
                    store=store,
                ),
            }
        store.put_cached_inspection(key, data)
        return ToolResponse(
            status="ok",
            data=data,
            provenance=_provenance(arguments, resolved_columns, store),
        )


def _resolve_columns(
    arguments: InspectHistoryArguments, store: RunStore
) -> tuple[list[str], list[ToolError]]:
    if arguments.view == "profiles":
        if not arguments.columns:
            return list(store.schema.names), []
        return list(arguments.columns), _validate_named_columns(arguments.columns, store, rows_view=False)

    if arguments.max_examples < 1:
        return [], [
            ToolError(
                code="rows_requires_examples",
                path="max_examples",
                message="rows inspection requires one-to-three sample rows.",
            )
        ]
    if not arguments.columns:
        return _all_rows_columns(store), []
    return list(arguments.columns), _validate_named_columns(arguments.columns, store, rows_view=True)


def _validate_named_columns(
    columns: list[str], store: RunStore, *, rows_view: bool
) -> list[ToolError]:
    errors: list[ToolError] = []
    seen: set[str] = set()
    for index, column in enumerate(columns):
        path = f"columns.{index}"
        if column in seen:
            errors.append(ToolError(code="duplicate_inspection_column", path=path, message="Columns must be unique."))
            continue
        seen.add(column)
        if column not in store.schema.names:
            errors.append(
                ToolError(
                    code="unknown_inspection_column",
                    path=path,
                    message=f"{column!r} is not a history column.",
                    allowed_values=store.schema.names,
                )
            )
            continue
        if rows_view:
            spec = store.schema.by_name(column)
            if spec.is_identifier or spec.kind not in {ColumnKind.NUMERIC, ColumnKind.CATEGORICAL}:
                errors.append(
                    ToolError(
                        code="rows_ineligible_column",
                        path=path,
                        message="rows inspection accepts only non-identifier numeric or categorical columns.",
                    )
                )
    return errors


def _all_rows_columns(store: RunStore) -> list[str]:
    return [
        spec.name
        for spec in store.schema.history_columns
        if not spec.is_identifier and spec.kind in {ColumnKind.NUMERIC, ColumnKind.CATEGORICAL}
    ]


def _profile_column(name: str, *, max_examples: int, store: RunStore) -> dict[str, object]:
    spec = store.schema.by_name(name)
    series = store.history[name]
    non_null = series.dropna()
    profile: dict[str, object] = {
        "column": name,
        "kind": spec.kind.value,
        "non_null_count": int(len(non_null)),
        "missing_count": int(series.isna().sum()),
        "unique_count": int(non_null.nunique(dropna=True)),
    }
    if spec.is_identifier:
        profile["is_identifier"] = True
        return profile
    if spec.kind == ColumnKind.NUMERIC:
        numeric = store.get_numeric_profile(name)
        profile["summary"] = {
            "min": numeric.minimum,
            "q05": numeric.q05,
            "median": numeric.median,
            "q95": numeric.q95,
            "max": numeric.maximum,
        }
        if max_examples:
            profile["representative_values"] = _numeric_examples(series, max_examples)
        if 0 < profile["unique_count"] <= MAX_ENUMERATED_NUMERIC_VALUES:
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            profile["observed_values"] = sorted(float(value) for value in values.unique())
        return profile
    if spec.kind == ColumnKind.CATEGORICAL:
        counts = non_null.astype(str).value_counts(dropna=True)
        profile["top_values"] = [
            {"value": value, "count": int(count)} for value, count in counts.head(10).items()
        ]
        if max_examples:
            profile["representative_values"] = counts.head(max_examples).index.tolist()
        return profile
    profile["summary"] = _non_tabular_summary(series, spec.kind)
    return profile


def _numeric_examples(series: pd.Series, maximum: int) -> list[float]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return []
    quantiles = np.linspace(0.0, 1.0, maximum)
    return [float(value) for value in np.quantile(numeric.to_numpy(dtype=float), quantiles)]


def _non_tabular_summary(series: pd.Series, kind: ColumnKind) -> dict[str, object]:
    non_null = series.dropna()
    if kind == ColumnKind.DATETIME:
        parsed = pd.to_datetime(non_null, errors="coerce").dropna()
        return {
            "min": parsed.min().isoformat() if not parsed.empty else None,
            "max": parsed.max().isoformat() if not parsed.empty else None,
        }
    lengths = non_null.astype(str).str.len()
    return {
        "median_length": float(lengths.median()) if not lengths.empty else None,
        "max_length": int(lengths.max()) if not lengths.empty else None,
    }


def _example_rows(*, columns: list[str], max_examples: int, store: RunStore) -> list[dict[str, object]]:
    rows = store.history.loc[:, columns].sample(
        n=min(max_examples, len(store.history)),
        random_state=store.random_state,
    )
    return [
        {name: _json_scalar(value) for name, value in record.items()}
        for record in rows.to_dict(orient="records")
    ]


def _json_scalar(value: object) -> object:
    return None if pd.isna(value) else value.item() if hasattr(value, "item") else value


def _provenance(
    arguments: InspectHistoryArguments, resolved_columns: list[str], store: RunStore
) -> dict[str, object]:
    excluded = sorted(set(store.schema.names) - set(_all_rows_columns(store)))
    return {
        "view": arguments.view,
        "requested_columns": arguments.columns,
        "resolved_columns": resolved_columns,
        "returned_example_count": arguments.max_examples,
        "history_rows_profiled": len(store.history),
        "random_seed": store.random_state,
        "source": "cleaned_history",
        "max_enumerated_numeric_values": MAX_ENUMERATED_NUMERIC_VALUES,
        "rows_view_excluded_columns": excluded if arguments.view == "rows" else [],
    }
