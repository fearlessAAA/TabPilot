"""Physical validation and sparse context preparation for the TFM.

This tool intentionally validates only executable table constraints.  Whether a
row faithfully represents the natural-language request is measured after the
run by evaluation-only grounding diagnostics, never decided by a runtime gate.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Any

import pandas as pd

from tabpilot.agents.prompts import (
    AMBIGUOUS_CATEGORY_FEEDBACK,
    CATEGORY_WITHOUT_LABELED_HISTORY_FEEDBACK,
    DEFERRED_CATEGORY_REPAIR_FEEDBACK,
    PREPARE_TFM_INPUT_TOOL_DESCRIPTION,
    PROVIDED_CASES_IGNORED_WARNING,
    PROVIDED_TARGET_MISSING_FEEDBACK,
    ROWS_REQUIRED_FEEDBACK,
    TFM_DTYPE_CONTRACT_FEEDBACK,
    UNSUPPORTED_CATEGORY_FEEDBACK,
    feature_excluded_from_model_message,
)
from tabpilot.tfm.context import (
    CategoryUnavailableForTargetError,
    FeatureCoverageCapacityError,
    TabPFNFrameContractError,
    build_model_context,
    model_excluded_columns,
    native_datetime_columns,
    to_utc_datetime,
)
from tabpilot.contracts import (
    ColumnKind,
    PredictionRow,
    PredictionType,
    PrepareTFMInputArguments,
    ToolError,
    ToolResponse,
)
from tabpilot.tools.category_candidates import retrieve_category_candidates
from tabpilot.tools.run_store import RunStore
from tabpilot.tools.canonicalization import canonical_column_name


class PrepareTFMInputTool:
    name = "prepare_tfm_input"
    description = PREPARE_TFM_INPUT_TOOL_DESCRIPTION
    input_model = PrepareTFMInputArguments

    def invoke(self, arguments: PrepareTFMInputArguments, store: RunStore) -> ToolResponse:
        if store.current is None and not arguments.rows:
            return ToolResponse(
                status="invalid",
                errors=[ToolError(code="rows_required", path="rows", message=ROWS_REQUIRED_FEEDBACK)],
            )
        provided_rows_ignored = store.current is not None and bool(arguments.rows)
        effective_arguments = arguments.model_copy(update={"rows": []}) if store.current is not None else arguments
        errors, target_column, normalized_rows, warnings = _validate_request(effective_arguments, store)
        if errors:
            return ToolResponse(status="invalid", errors=errors)
        if provided_rows_ignored:
            warnings.append(
                {
                    "code": "provided_rows_ignored",
                    "submitted_row_count": len(arguments.rows),
                    "message": PROVIDED_CASES_IGNORED_WARNING,
                }
            )
        if store.current is not None and target_column in store.current.columns:
            if bool(store.current[target_column].notna().any()):
                # Do not reveal values or counts from a populated candidate outcome.
                return ToolResponse(
                    status="invalid",
                    errors=[
                        ToolError(
                            code="provided_target_must_be_missing",
                            path="target_column",
                            message=PROVIDED_TARGET_MISSING_FEEDBACK,
                            retryable=False,
                        )
                    ],
                )
        type_evidence, type_warnings, type_error = _validate_prediction_type(
            store.history, target_column, effective_arguments.prediction_type
        )
        if type_error is not None:
            return ToolResponse(status="invalid", errors=[type_error])
        warnings.extend(type_warnings)
        try:
            provided_rows = store.current is not None
            runtime_rows = (
                _rows_from_provided_table(store.current, store.schema)
                if store.current is not None
                else normalized_rows
            )
            context = build_model_context(
                history=store.history,
                schema=store.schema,
                target_column=target_column,
                rows=runtime_rows,
                query_frame=store.current if provided_rows else None,
                query_row_ids=[row.row_id for row in runtime_rows] if provided_rows else None,
                max_rows=store.max_context_rows,
                max_features=store.max_context_features,
                random_state=store.random_state,
                model_features=store.model_features,
            )
        except FeatureCoverageCapacityError as error:
            return ToolResponse(
                status="invalid",
                errors=[
                    ToolError(
                        code="feature_coverage_capacity_exceeded",
                        message=str(error),
                    )
                ],
            )
        except CategoryUnavailableForTargetError as error:
            return ToolResponse(
                status="invalid",
                errors=[
                    ToolError(
                        code="category_without_labeled_history",
                        path=f"rows.{error.row_index}.features.{error.feature}",
                        message=CATEGORY_WITHOUT_LABELED_HISTORY_FEEDBACK,
                    )
                ],
            )
        except TabPFNFrameContractError:
            return ToolResponse(
                status="error",
                errors=[
                    ToolError(
                        code="tfm_dtype_contract",
                        message=TFM_DTYPE_CONTRACT_FEEDBACK,
                        retryable=False,
                    )
                ],
            )
        except ValueError as error:
            return ToolResponse(
                status="invalid",
                errors=[ToolError(code="context_build_failed", message=str(error))],
            )
        artifact = store.put_prepared(
            target_column=target_column,
            prediction_type=effective_arguments.prediction_type,
            rows=runtime_rows,
            context=context,
            type_evidence=type_evidence,
            warnings=warnings,
        )
        # A supplied current table is runtime-only input.  In particular, do
        # not echo its runtime-expanded rows into the tool observation: the
        # next LLM turn receives that observation and must never see table
        # values (nor should a large table consume the chat context).
        data: dict[str, object] = {
                "prepared_id": artifact.prepared_id,
                "target_column": target_column,
                "requested_prediction_type": effective_arguments.prediction_type,
                "resolved_prediction_type": effective_arguments.prediction_type,
                "type_evidence": type_evidence,
                "query_source": "provided_cases_table" if provided_rows else "explicit_rows",
                "feature_columns": context.feature_columns,
                "excluded_columns": context.excluded_columns,
                "missing_strategy": context.missing_strategy,
                "category_coverage_constraints": context.category_coverage_constraints,
                "query_missing_counts": context.query_missing_counts,
                "warnings": warnings,
        }
        if provided_rows:
            data["provided_row_count"] = len(runtime_rows)
            data["provided_unrepresentable_category_cell_count"] = len(
                context.provided_unrepresentable_category_cells
            )
        else:
            data["normalized_rows"] = [row.model_dump(mode="json") for row in runtime_rows]
        provenance = {
            "history_rows_used": context.history_rows_used,
            "feature_count": len(context.feature_columns),
            "categorical_features_indices": context.categorical_features_indices,
            "history_feature_missing_count": context.history_feature_missing_count,
            "sampling_strategy": context.sampling_strategy,
            "sampling_seed": context.sampling_seed,
            "sampled_row_hash": context.sampled_row_hash,
            "missing_strategy": context.missing_strategy,
            "category_coverage_constraints": context.category_coverage_constraints,
        }
        if provided_rows:
            # Trace-only diagnostics of the provided table.
            provenance.update(
                {
                    "provided_unrepresentable_category_cell_count": len(
                        context.provided_unrepresentable_category_cells
                    ),
                    "provided_unrepresentable_category_reasons": dict(
                        Counter(cell.reason for cell in context.provided_unrepresentable_category_cells)
                    ),
                    "normalized_rows": [],
                }
            )
        return ToolResponse(status="ok", data=data, provenance=provenance)


def _validate_request(
    arguments: PrepareTFMInputArguments, store: RunStore
) -> tuple[list[ToolError], str, list[PredictionRow], list[dict[str, object]]]:
    errors: list[ToolError] = []
    warnings: list[dict[str, object]] = []
    schema = store.schema
    names = set(schema.names)
    target, target_status = canonical_column_name(arguments.target_column, schema.names)
    if target is None:
        target = arguments.target_column
    if target not in names:
        errors.append(ToolError(code="unknown_target", path="target_column", message=f"{target!r} is not a history column.", allowed_values=schema.names))
        return errors, target, [], warnings
    if schema.by_name(target).is_identifier:
        errors.append(ToolError(code="identifier_target", path="target_column", message="Identifier-like columns are not prediction targets."))
    if target_status == "canonicalized":
        warnings.append(
            {"code": "column_name_canonicalized", "path": "target_column", "canonical_value": target}
        )
    native_dates = native_datetime_columns(store.history, schema)
    policy_excluded = model_excluded_columns(store.history, schema, store.model_features)
    model_kinds = {ColumnKind.NUMERIC, ColumnKind.CATEGORICAL, ColumnKind.OPEN_TEXT, ColumnKind.DATETIME}
    target_spec = schema.by_name(target)
    if target_spec.kind in {ColumnKind.OPEN_TEXT, ColumnKind.DATETIME}:
        errors.append(ToolError(code="unsupported_target_type", path="target_column", message=f"Target kind {target_spec.kind.value!r} is not supported by the tabular foundation model."))

    seen_ids: set[str] = set()
    normalized_rows: list[PredictionRow] = []
    for row_index, row in enumerate(arguments.rows):
        prefix = f"rows.{row_index}"
        if row.row_id in seen_ids:
            errors.append(
                ToolError(
                    code="duplicate_row_id",
                    path=f"{prefix}.row_id",
                    message="row_id must be unique.",
                )
            )
        seen_ids.add(row.row_id)
        normalized_features: dict[str, Any] = {}
        for feature, value in row.features.items():
            feature_path = f"{prefix}.features.{feature}"
            if feature == target:
                errors.append(ToolError(code="target_in_features", path=feature_path, message="The target column cannot also be a query feature."))
                continue
            canonical_feature, feature_status = canonical_column_name(feature, schema.names)
            if canonical_feature is None:
                errors.append(ToolError(code="unknown_feature", path=feature_path, message=f"{feature!r} is not a history column.", allowed_values=schema.names))
                continue
            if feature_status == "canonicalized":
                warnings.append(
                    {
                        "code": "column_name_canonicalized",
                        "path": feature_path,
                        "canonical_value": canonical_feature,
                    }
                )
            feature = canonical_feature
            feature_path = f"{prefix}.features.{feature}"
            spec = schema.by_name(feature)
            scalar_kind = spec.kind
            if spec.is_identifier and spec.kind == ColumnKind.CATEGORICAL:
                # An ID can be novel at inference time. Its historical category list is not
                # an admissibility rule: numeric IDs stay numeric; string IDs become text.
                scalar_kind = (
                    ColumnKind.NUMERIC
                    if pd.api.types.is_numeric_dtype(store.history[feature])
                    else ColumnKind.OPEN_TEXT
                )
            normalized, scalar_warnings, scalar_error = _validate_scalar(
                value,
                scalar_kind,
                spec.allowed_values if scalar_kind == ColumnKind.CATEGORICAL else [],
                feature_path,
                (
                    _labeled_supported_values(store.history, target, feature)
                    if scalar_kind == ColumnKind.CATEGORICAL else []
                ),
            )
            if scalar_error is not None:
                errors.append(scalar_error)
                continue
            if normalized is _INVALID:
                errors.append(ToolError(code="invalid_numeric", path=feature_path, message="Numeric values must be finite real numbers."))
                continue
            if (
                feature in native_dates
                and normalized is not None
                and pd.isna(to_utc_datetime(pd.Series([normalized])).iloc[0])
            ):
                errors.append(ToolError(code="invalid_datetime", path=feature_path, message=f"{feature!r} is a date column; provide a parseable date."))
                continue
            normalized_features[feature] = normalized
            warnings.extend(scalar_warnings)
            if feature in policy_excluded:
                warnings.append(
                    {
                        "code": "feature_excluded_from_model",
                        "path": feature_path,
                        "message": feature_excluded_from_model_message(feature, spec.kind.value),
                    }
                )
        omitted = [
            name for name in schema.names
            if name != target
            and schema.by_name(name).kind in model_kinds
            and name not in normalized_features
            and name not in policy_excluded
        ]
        if omitted:
            warnings.append({"code": "unstated_features_missing", "path": f"{prefix}.features", "message": "Unstated model features remain native missing values (NaN).", "columns": omitted})
        normalized_rows.append(PredictionRow(row_id=row.row_id, features=normalized_features))
    return errors, target, normalized_rows, warnings
def _rows_from_provided_table(current: pd.DataFrame, schema: object) -> list[PredictionRow]:
    """Create trace bookkeeping rows while preserving ``current`` as the query frame.

    The returned dictionaries are never used to construct the TabPFN DataFrame.
    """
    safe_schema = schema
    eligible = [
        column.name
        for column in safe_schema.history_columns
        if column.name in current.columns
        and not column.is_identifier
        and column.kind in {ColumnKind.NUMERIC, ColumnKind.CATEGORICAL}
    ]
    rows: list[PredictionRow] = []
    for index in range(len(current)):
        source = current.iloc[index]
        features = {
            column: _json_scalar(source[column])
            for column in eligible
            if pd.notna(source[column])
        }
        rows.append(PredictionRow(row_id=f"case_{index + 1}", features=features))
    return rows


def _json_scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


_INVALID = object()


def _normalise_category(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()


def _numeric_category_equivalent(left: object, right: object) -> bool:
    """Match JSON numeric spellings to the one canonical SafeSchema category.

    Category codes are discrete, so this deliberately accepts only exact numeric
    equality after conversion (for example ``0`` and ``0.0``), never a tolerance
    that could merge distinct category identifiers. Booleans remain distinct
    categorical values even though Python treats them as integers.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isfinite(float(left)) and math.isfinite(float(right)) and float(left) == float(right)


def _validate_scalar(
    value: object,
    kind: ColumnKind,
    allowed_values: list[object],
    path: str,
    labeled_supported_values: list[object],
) -> tuple[object, list[dict[str, object]], ToolError | None]:
    if value is None:
        return None, [], None
    if kind == ColumnKind.NUMERIC:
        if isinstance(value, bool):
            return _INVALID, [], None
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _INVALID, [], None
        if not math.isfinite(number):
            return _INVALID, [], None
        normalized = int(number) if isinstance(value, int) and not isinstance(value, bool) else number
        return normalized, [], None
    if kind != ColumnKind.CATEGORICAL:
        normalized = str(value) if kind in {ColumnKind.OPEN_TEXT, ColumnKind.DATETIME} else value
        return normalized, [], None
    if not allowed_values:
        return (
            value,
            [],
            ToolError(
                code="category_domain_unavailable",
                path=path,
                message=UNSUPPORTED_CATEGORY_FEEDBACK,
                candidate_values=[],
            ),
        )
    numeric_matches = [
        allowed
        for allowed in allowed_values
        if _numeric_category_equivalent(allowed, value)
    ]
    if len(numeric_matches) == 1:
        return numeric_matches[0], [], None
    exact_matches = [
        allowed
        for allowed in allowed_values
        if type(allowed) is type(value) and allowed == value
    ]
    if len(exact_matches) == 1:
        return exact_matches[0], [], None
    matches = [allowed for allowed in allowed_values if _normalise_category(allowed) == _normalise_category(value)]
    if len(matches) == 1:
        canonical = matches[0]
        warning: list[dict[str, object]] = []
        if str(canonical) != str(value):
            warning.append({"code": "category_normalized", "path": path, "message": "Category matched an allowed value after case/whitespace normalization.", "canonical_value": canonical})
        return canonical, warning, None
    if len(matches) > 1:
        return (
            value,
            [],
            ToolError(
                code="ambiguous_category",
                path=path,
                message=AMBIGUOUS_CATEGORY_FEEDBACK,
                submitted_value=value,
                candidate_values=matches[:5],
            ),
        )
    return (
        value,
        [],
        ToolError(
            code="unsupported_category",
            path=path,
            message=DEFERRED_CATEGORY_REPAIR_FEEDBACK,
            submitted_value=value,
            candidate_values=retrieve_category_candidates(
                value,
                allowed_values,
                labeled_supported_values=labeled_supported_values,
            ),
        ),
    )


def _labeled_supported_values(history: pd.DataFrame, target: str, feature: str) -> list[object]:
    """Return diagnostic support from the agent-hypothesized target only.

    This is used solely to order equal lexical repair candidates. It does not
    filter the complete SafeSchema domain and therefore never changes semantic
    acceptance rules or reveals a benchmark target.
    """
    if target not in history.columns or feature not in history.columns:
        return []
    values = history.loc[history[target].notna(), feature].dropna().tolist()
    return [value for value in values if isinstance(value, (str, int, float, bool))]


def _validate_prediction_type(history: pd.DataFrame, target: str, requested: PredictionType) -> tuple[str, list[dict[str, object]], ToolError | None]:
    series = history[target].dropna()
    if series.empty:
        return "", [], ToolError(code="empty_target", path="target_column", message="Target has no labels.")
    if not pd.api.types.is_numeric_dtype(series):
        if requested == "regression":
            return "Target values are non-numeric.", [], ToolError(code="non_numeric_regression_target", path="prediction_type", message="Regression requires a numeric target column.")
        return "Target values are non-numeric.", [], None
    cardinality = int(series.nunique())
    warning: dict[str, object] = {
        "code": "numeric_target_cardinality",
        "path": "prediction_type",
        "message": "Numeric target representation is diagnostic only; the requested prediction type is unchanged.",
        "distinct_values": cardinality,
    }
    # This is a representation fact, not a recommendation.  It gives the
    # Agent a chance to reconsider a hypothesized numeric grade/band without
    # letting the tool decide its semantic prediction type.
    if 0 < cardinality <= 20:
        warning["observed_numeric_values"] = sorted(float(value) for value in series.unique())
    return (
        f"Target values are numeric with {cardinality} distinct non-null values.",
        [warning],
        None,
    )
