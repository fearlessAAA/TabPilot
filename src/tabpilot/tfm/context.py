"""Build raw, leakage-safe TabPFN contexts from sparse agent rows.

TabPilot deliberately does not encode categories or impute query values.  TabPFN
V2 receives the raw history/query ``DataFrame`` objects, categorical column
indices, and native ``NaN`` values so that its own preprocessing remains the
single source of truth for categorical and missing-value handling.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from tabpilot.contracts import ColumnKind, PredictionRow, SafeSchema

# A string column is treated as a date only when nearly all of its values parse.
_DATETIME_MIN_PARSE_SHARE = 0.95
_DATETIME_MAX_NUMERIC_SHARE = 0.5
_DATETIME_DETECTION_SAMPLE = 2000


class TabPFNFrameContractError(ValueError):
    """The raw prepared frames cannot pass TabPFN's fit/predict dtype contract."""


class CategoryUnavailableForTargetError(ValueError):
    """A legal schema category has no labeled history support for this target."""

    def __init__(self, *, row_index: int, feature: str, value: object) -> None:
        self.row_index = row_index
        self.feature = feature
        self.value = value
        super().__init__(
            f"categorical value {value!r} for {feature!r} has no labeled history support"
        )


class CategoryCoverageCapacityError(ValueError):
    """The bounded context cannot retain all required categorical witnesses."""


class FeatureCoverageCapacityError(ValueError):
    """The number of query-stated features exceeds the backend's feature-count ceiling."""


@dataclass(frozen=True)
class _CategoryRequirement:
    row_index: int
    feature: str
    value: object
    canonical_value: str | float


@dataclass(frozen=True)
class ProvidedUnrepresentableCategoryCell:
    """One runtime-provided categorical cell absent from labeled fit data.

    The record deliberately contains no category value.  It is provenance for a
    physical missing-value conversion, not model-visible information about the
    supplied candidate table.
    """

    row_index: int
    feature: str
    reason: Literal["seen_but_unlabeled", "unseen_in_history"]


@dataclass(frozen=True)
class ModelContext:
    x_history: pd.DataFrame
    y_history: pd.Series
    x_query: pd.DataFrame
    row_ids: list[str]
    feature_columns: list[str]
    categorical_features_indices: list[int]
    excluded_columns: list[str]
    query_missing_counts: dict[str, int]
    history_feature_missing_count: int
    sampled_row_hash: str
    history_rows_used: int
    sampling_strategy: str
    sampling_seed: int
    categorical_scalar_kinds: dict[str, Literal["numeric", "string"]]
    category_coverage_constraints: dict[str, int]
    provided_unrepresentable_category_cells: tuple[ProvidedUnrepresentableCategoryCell, ...] = ()
    missing_strategy: str = "native_nan"
    # DATETIME/OPEN_TEXT columns (and string identifiers) are passed to the TFM as datetime64/string
    # columns for its own date and text expansion.
    datetime_columns: tuple[str, ...] = ()
    text_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelFeatures:
    """Which column kinds reach the TFM; the agent-facing contract (prompts, validation) is unchanged.

    A TFM whose official preprocessing cannot use dates or free text turns them off here; such
    columns are reported in ``excluded_columns``. String identifiers are passed as text, so they
    also need ``text``; numeric identifiers need only ``identifiers``.
    """

    datetime: bool = True
    text: bool = True
    identifiers: bool = True


def model_excluded_columns(
    history: pd.DataFrame, schema: SafeSchema, model_features: ModelFeatures | None
) -> frozenset[str]:
    """Feature columns that ``model_features`` keeps out of the TFM (shared with ``prepare_tfm_input``)."""
    if model_features is None or model_features == ModelFeatures():
        return frozenset()
    native_dates = native_datetime_columns(history, schema)
    excluded: set[str] = set()
    for spec in schema.history_columns:
        if spec.name not in history.columns:
            continue
        needs: set[str] = set()
        if spec.is_identifier:
            needs.add("identifiers")
        if spec.kind in {ColumnKind.DATETIME, ColumnKind.OPEN_TEXT}:
            needs.add("datetime" if spec.name in native_dates else "text")
        elif spec.is_identifier and spec.kind == ColumnKind.CATEGORICAL and not pd.api.types.is_numeric_dtype(
            history[spec.name]
        ):
            needs.add("text")
        if any(not getattr(model_features, kind) for kind in needs):
            excluded.add(spec.name)
    return frozenset(excluded)


def build_model_context(
    *,
    history: pd.DataFrame,
    schema: SafeSchema,
    target_column: str,
    rows: list[PredictionRow],
    query_frame: pd.DataFrame | None = None,
    query_row_ids: list[str] | None = None,
    max_rows: int = 10_000,
    max_features: int | None = None,
    random_state: int = 42,
    model_features: ModelFeatures | None = None,
) -> ModelContext:
    """Return raw history/query data suitable for the selected TFM.

    Query values absent from ``PredictionRow.features`` remain ``np.nan``.  We
    never fit a transform on query rows, fill missing values, or expose any
    evaluation-only information at this boundary.
    """
    eligible_full = history.dropna(subset=[target_column]).copy()
    if eligible_full.empty:
        raise ValueError(f"history contains no non-null target values for {target_column!r}")

    specs = {column.name: column for column in schema.history_columns}
    # Every column type is a TFM feature, identifiers included.
    modality_kinds = {ColumnKind.NUMERIC, ColumnKind.CATEGORICAL, ColumnKind.DATETIME, ColumnKind.OPEN_TEXT}
    feature_columns = [
        name
        for name in eligible_full.columns
        if name != target_column and specs[name].kind in modality_kinds
    ]
    excluded_columns = [
        name
        for name in eligible_full.columns
        if name != target_column and specs[name].kind not in modality_kinds
    ]
    policy_excluded = model_excluded_columns(eligible_full, schema, model_features)
    excluded_columns += [name for name in feature_columns if name in policy_excluded]
    feature_columns = [name for name in feature_columns if name not in policy_excluded]
    if not feature_columns:
        raise ValueError("no TabPFN-compatible numeric or categorical feature columns remain")

    feature_columns = _cap_feature_columns(
        feature_columns,
        rows=rows,
        query_frame=query_frame,
        max_features=max_features,
        random_state=random_state,
    )

    # Identifier strings have an open vocabulary: an unseen query ID is valid, but the
    # historical categorical-domain validator would reject it. Let TabPFN encode them as text.
    identifier_text_features = [
        name for name in feature_columns
        if specs[name].is_identifier
        and specs[name].kind == ColumnKind.CATEGORICAL
        and not pd.api.types.is_numeric_dtype(eligible_full[name])
    ]
    identifier_numeric_features = [
        name for name in feature_columns
        if specs[name].is_identifier
        and specs[name].kind == ColumnKind.CATEGORICAL
        and pd.api.types.is_numeric_dtype(eligible_full[name])
    ]
    categorical_features = [
        name for name in feature_columns
        if specs[name].kind == ColumnKind.CATEGORICAL
        and name not in identifier_text_features
        and name not in identifier_numeric_features
    ]
    categorical_scalar_kinds = {
        name: _categorical_scalar_kind(eligible_full[name]) for name in categorical_features
    }
    provided_unrepresentable_category_cells: tuple[ProvidedUnrepresentableCategoryCell, ...] = ()
    if query_frame is None:
        requirements = _required_categorical_values(rows, categorical_scalar_kinds)
    else:
        requirements = _required_categorical_values_from_frame(
            query_frame,
            categorical_scalar_kinds,
        )
        (
            requirements,
            provided_unrepresentable_category_cells,
        ) = _partition_provided_category_requirements(
            requirements=requirements,
            query_frame=query_frame,
            eligible=eligible_full,
            schema=schema,
            scalar_kinds=categorical_scalar_kinds,
        )
    eligible, sampling_strategy, coverage_constraints = _sample_labeled_history(
        eligible_full,
        requirements=requirements,
        max_rows=max_rows,
        random_state=random_state,
    )

    x_history = eligible.loc[:, feature_columns].copy()
    # Start from the selected history's column layout, not a generic object
    # frame.  Per-column coercion below completes the raw TabPFN frame
    # contract while keeping all unstated facts as native missing values.
    if query_frame is None:
        x_query = pd.DataFrame(index=range(len(rows)), columns=feature_columns)
        for feature in feature_columns:
            # Object construction preserves a supplied scalar such as ``0`` rather
            # than allowing Pandas to promote ``[0, NaN]`` to ``0.0`` before the
            # history-derived column contract decides whether it is numeric or a
            # string category.
            x_query[feature] = pd.Series(
                [row.features.get(feature, np.nan) for row in rows],
                index=x_query.index,
                dtype="object",
            )
    else:
        missing = [name for name in feature_columns if name not in query_frame.columns]
        if missing:
            raise TabPFNFrameContractError(
                f"provided query frame lacks history feature columns: {', '.join(missing)}"
            )
        # Keep the provided DataFrame as the source of query scalar values.  In
        # particular, do not reconstruct it from JSON-like PredictionRow dicts:
        # nullable integers and categoricals would lose their fit-time semantics.
        x_query = query_frame.loc[:, feature_columns].copy()
    numeric_features = [
        name for name in feature_columns
        if specs[name].kind == ColumnKind.NUMERIC or name in identifier_numeric_features
    ]

    # This is type preservation, not imputation. ``raise`` guarantees a value
    # accepted by prepare cannot silently become missing while building context.
    # A sparse query with native ``NaN`` cannot retain a NumPy integer dtype.
    # Canonicalize both sides of every numeric column to float64 so TabPFN sees
    # identical fitted/prediction dtypes while retaining the original values and
    # native missingness. This is neither scaling nor imputation.
    for name in numeric_features:
        x_history[name] = pd.to_numeric(x_history[name], errors="coerce").astype("float64")
        source = x_query[name]
        converted = pd.to_numeric(source, errors="raise").astype("float64")
        if bool(source.notna().any() and converted[source.notna()].isna().any()):
            raise ValueError(f"numeric query value for {name!r} could not be preserved")
        x_query[name] = converted
    expected_provided_losses = {
        (cell.row_index, cell.feature) for cell in provided_unrepresentable_category_cells
    }
    original_categorical_non_null = {
        (row_index, name)
        for name in categorical_features
        for row_index, value in enumerate(x_query[name].tolist())
        if pd.notna(value)
    }
    for row_index, name in expected_provided_losses:
        x_query.at[x_query.index[row_index], name] = np.nan

    for name in categorical_features:
        # Derive the scalar representation from history only. Merely casting both
        # sides to ``object`` is insufficient: an object history column can hold
        # strings while an object query column holds integers, which makes
        # TabPFN's fitted ordinal encoder see a predict-time dtype drift.
        scalar_kind = categorical_scalar_kinds[name]
        x_history[name], x_query[name] = _align_categorical_column(
            x_history[name],
            x_query[name],
            name=name,
            scalar_kind=scalar_kind,
        )

    lost_categorical_cells = {
        (row_index, name)
        for row_index, name in original_categorical_non_null
        if pd.isna(x_query.iloc[row_index][name])
    }
    if lost_categorical_cells != expected_provided_losses:
        raise TabPFNFrameContractError(
            "categorical query values were lost outside the exact provided-table allowlist"
        )

    datetime_columns: list[str] = []
    text_columns: list[str] = []
    native_dates = native_datetime_columns(eligible_full, schema)
    for name in feature_columns:
        if specs[name].kind not in {ColumnKind.DATETIME, ColumnKind.OPEN_TEXT} \
                and name not in identifier_text_features:
            continue
        if name in native_dates:
            datetime_columns.append(name)
            x_history[name] = to_utc_datetime(x_history[name])
            source = x_query[name]
            converted = to_utc_datetime(source)
            if bool((source.notna() & converted.isna()).any()):
                raise ValueError(f"datetime query value for {name!r} could not be parsed")
            x_query[name] = converted
        else:
            text_columns.append(name)
            x_history[name] = _as_text(x_history[name])
            x_query[name] = _as_text(x_query[name])

    _assert_tabpfn_preprocessing_contract(
        **_contract_view(x_history, x_query, feature_columns, categorical_features, datetime_columns + text_columns)
    )

    sampled_row_hash = hashlib.sha256(",".join(map(str, eligible.index.tolist())).encode()).hexdigest()
    return ModelContext(
        x_history=x_history.reset_index(drop=True),
        y_history=eligible[target_column].reset_index(drop=True),
        x_query=x_query.reset_index(drop=True),
        row_ids=query_row_ids or [row.row_id for row in rows],
        feature_columns=feature_columns,
        categorical_features_indices=[feature_columns.index(name) for name in categorical_features],
        excluded_columns=excluded_columns,
        query_missing_counts={name: int(x_query[name].isna().sum()) for name in feature_columns},
        history_feature_missing_count=int(x_history.isna().sum().sum()),
        sampled_row_hash=sampled_row_hash,
        history_rows_used=len(eligible),
        sampling_strategy=sampling_strategy,
        sampling_seed=random_state,
        categorical_scalar_kinds=categorical_scalar_kinds,
        category_coverage_constraints=coverage_constraints,
        provided_unrepresentable_category_cells=provided_unrepresentable_category_cells,
        datetime_columns=tuple(datetime_columns),
        text_columns=tuple(text_columns),
    )


def native_datetime_columns(history: pd.DataFrame, schema: SafeSchema) -> frozenset[str]:
    """DATETIME/OPEN_TEXT columns passed to the native predictor as dates.

    A DATETIME column always qualifies. An OPEN_TEXT column qualifies only when, on a deterministic
    sample of its non-missing values, at most half parse as numbers and at least 95% parse as dates;
    every other OPEN_TEXT column is passed as text. Shared by ``prepare_tfm_input`` validation and
    context construction so both sides agree.
    """
    dates: set[str] = set()
    for column in schema.history_columns:
        if column.kind not in {ColumnKind.DATETIME, ColumnKind.OPEN_TEXT}:
            continue
        if column.name not in history.columns:
            continue
        if column.kind == ColumnKind.DATETIME:
            dates.add(column.name)
            continue
        values = history[column.name].dropna()
        if values.empty:
            continue
        if len(values) > _DATETIME_DETECTION_SAMPLE:
            values = values.sample(_DATETIME_DETECTION_SAMPLE, random_state=0)
        text = values.astype("string")
        if pd.to_numeric(text, errors="coerce").notna().mean() > _DATETIME_MAX_NUMERIC_SHARE:
            continue
        if to_utc_datetime(text).notna().mean() >= _DATETIME_MIN_PARSE_SHARE:
            dates.add(column.name)
    return frozenset(dates)


def to_utc_datetime(values: pd.Series) -> pd.Series:
    """Parse to tz-aware UTC datetimes; unparseable values become ``NaT``.

    A single inferred format is tried first (vectorized); columns mixing formats fall back to
    per-value parsing.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, utc=True)
    text = values.astype("string")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            parsed = pd.to_datetime(text, errors="coerce", utc=True)
        except (ValueError, TypeError, OverflowError):
            parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns, UTC]")
        if parsed[text.notna()].isna().any():
            try:
                mixed = pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
            except (ValueError, TypeError, OverflowError):
                mixed = parsed
            if mixed.notna().sum() > parsed.notna().sum():
                parsed = mixed
    return parsed


def _as_text(values: pd.Series) -> pd.Series:
    """String dtype the predictor's text detection accepts; missing values stay missing.

    TabPFN expands only pandas' ``string`` extension dtype as text. ``astype("str")`` gives that
    dtype on pandas 3 but plain ``object`` on pandas 2, where text columns would silently be
    ordinal-encoded instead; spell the pandas-3 ``str`` dtype out so both versions agree.
    """
    return values.map(lambda value: value if pd.isna(value) else str(value)).astype(
        pd.StringDtype(na_value=np.nan)
    )


def _contract_view(
    x_history: pd.DataFrame,
    x_query: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
    native_columns: list[str],
) -> dict[str, object]:
    """The numeric/categorical sub-frame checked by the TabPFN ordinal-encoding contract."""
    kept = [name for name in feature_columns if name not in set(native_columns)]
    return {
        "x_history": x_history.loc[:, kept],
        "x_query": x_query.loc[:, kept],
        "categorical_features_indices": [kept.index(name) for name in categorical_features],
    }


def _required_categorical_values_from_frame(
    frame: pd.DataFrame,
    scalar_kinds: dict[str, Literal["numeric", "string"]],
) -> list[_CategoryRequirement]:
    """Collect fitted-vocabulary witnesses directly from a bound DataFrame.

    This deliberately never serializes table values through PredictionRow.
    """
    requirements: list[_CategoryRequirement] = []
    seen: set[tuple[str, str | float]] = set()
    for feature, scalar_kind in scalar_kinds.items():
        if feature not in frame.columns:
            continue
        for row_index, value in enumerate(frame[feature].tolist()):
            if pd.isna(value):
                continue
            canonical_value = _canonical_categorical_value(value, scalar_kind)
            key = (feature, canonical_value)
            if key in seen:
                continue
            seen.add(key)
            requirements.append(
                _CategoryRequirement(
                    row_index=row_index,
                    feature=feature,
                    value=value,
                    canonical_value=canonical_value,
                )
            )
    return requirements


def _required_categorical_values(
    rows: list[PredictionRow],
    scalar_kinds: dict[str, Literal["numeric", "string"]],
) -> list[_CategoryRequirement]:
    """Collect stated categorical values that the fitted history must retain.

    This is a physical vocabulary-closure requirement, not a semantic context
    selection policy: only values the agent already submitted and SafeSchema
    validation already accepted can become requirements.
    """
    requirements: list[_CategoryRequirement] = []
    seen: set[tuple[str, str | float]] = set()
    for row_index, row in enumerate(rows):
        for feature, scalar_kind in scalar_kinds.items():
            value = row.features.get(feature)
            if value is None:
                continue
            try:
                canonical_value = _canonical_categorical_value(value, scalar_kind)
            except (TypeError, ValueError) as error:
                # A category may appear in the full cleaned table yet have no
                # compatible representation among rows with an observed target.
                # Treat that as absent labeled support, never as a dtype-system
                # failure or an invitation to coerce the value to unknown.
                raise CategoryUnavailableForTargetError(
                    row_index=row_index,
                    feature=feature,
                    value=value,
                ) from error
            key = (feature, canonical_value)
            if key in seen:
                continue
            seen.add(key)
            requirements.append(
                _CategoryRequirement(
                    row_index=row_index,
                    feature=feature,
                    value=value,
                    canonical_value=canonical_value,
                )
            )
    return requirements


def _partition_provided_category_requirements(
    *,
    requirements: list[_CategoryRequirement],
    query_frame: pd.DataFrame,
    eligible: pd.DataFrame,
    schema: SafeSchema,
    scalar_kinds: dict[str, Literal["numeric", "string"]],
) -> tuple[list[_CategoryRequirement], tuple[ProvidedUnrepresentableCategoryCell, ...]]:
    """Separate provided categories with fit witnesses from legal no-witness cells.

    A runtime-bound table is not model-authored: its values cannot be repaired by
    the LLM.  A value that belongs to the complete SafeSchema vocabulary but has
    no labeled row for the selected target is therefore represented as native
    missing in *that exact query cell*.  Explicit rows never enter this path.
    """
    supported_keys: set[tuple[str, str | float]] = set()
    missing_reasons: dict[tuple[str, str | float], Literal["seen_but_unlabeled", "unseen_in_history"]] = {}
    for requirement in requirements:
        key = (requirement.feature, requirement.canonical_value)
        if not _safe_schema_contains_category(
            schema.by_name(requirement.feature).allowed_values,
            requirement.canonical_value,
            scalar_kinds[requirement.feature],
        ):
            # The value originated from the runtime-bound candidate table, not
            # an LLM row. TabPFN has no fitted category for it, so native
            # missing is the only faithful raw representation.
            missing_reasons[key] = "unseen_in_history"
            continue
        scalar_kind = scalar_kinds[requirement.feature]
        values = _canonical_categorical_series(eligible[requirement.feature], scalar_kind)
        if bool(values.eq(requirement.canonical_value).any()):
            supported_keys.add(key)
        else:
            missing_reasons[key] = "seen_but_unlabeled"

    cells: list[ProvidedUnrepresentableCategoryCell] = []
    for feature, scalar_kind in scalar_kinds.items():
        if feature not in query_frame.columns:
            continue
        for row_index, value in enumerate(query_frame[feature].tolist()):
            if pd.isna(value):
                continue
            canonical_value = _canonical_categorical_value(value, scalar_kind)
            reason = missing_reasons.get((feature, canonical_value))
            if reason is not None:
                cells.append(
                    ProvidedUnrepresentableCategoryCell(
                        row_index=row_index,
                        feature=feature,
                        reason=reason,
                    )
                )
    return (
        [
            requirement
            for requirement in requirements
            if (requirement.feature, requirement.canonical_value) in supported_keys
        ],
        tuple(cells),
    )


def _safe_schema_contains_category(
    allowed_values: list[object],
    canonical_value: str | float,
    scalar_kind: Literal["numeric", "string"],
) -> bool:
    """Check closed-vocabulary membership with the history-derived scalar form."""
    for value in allowed_values:
        try:
            if _canonical_categorical_value(value, scalar_kind) == canonical_value:
                return True
        except (TypeError, ValueError, TabPFNFrameContractError):
            continue
    return False


def _cap_feature_columns(
    feature_columns: list[str],
    *,
    rows: list[PredictionRow],
    query_frame: pd.DataFrame | None,
    max_features: int | None,
    random_state: int,
) -> list[str]:
    """Bound the feature matrix width while retaining every query-stated feature.

    A stated feature (one with a non-null value somewhere in the query) is a
    witness the model must keep, mirroring ``_sample_labeled_history``'s
    row-witness policy: if not exceeding the cap, keep everything; if
    exceeding it, keep every witness and fill the remainder with a
    deterministic random sample of the unstated columns.
    """
    if max_features is None or len(feature_columns) <= max_features:
        return feature_columns
    if query_frame is None:
        witnesses = {
            feature
            for row in rows
            for feature, value in row.features.items()
            if value is not None and feature in feature_columns
        }
    else:
        witnesses = {
            feature
            for feature in feature_columns
            if feature in query_frame.columns and bool(query_frame[feature].notna().any())
        }
    if len(witnesses) > max_features:
        raise FeatureCoverageCapacityError(
            "The number of query-stated feature columns exceeds the predictor's feature-count limit."
        )
    remaining = [name for name in feature_columns if name not in witnesses]
    fill_count = max_features - len(witnesses)
    rng = np.random.default_rng(random_state)
    sampled_remaining = (
        list(rng.choice(remaining, size=fill_count, replace=False)) if fill_count > 0 else []
    )
    kept = {name for name in feature_columns if name in witnesses or name in sampled_remaining}
    return [name for name in feature_columns if name in kept]


def _sample_labeled_history(
    eligible: pd.DataFrame,
    *,
    requirements: list[_CategoryRequirement],
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, str, dict[str, int]]:
    """Sample labeled history while retaining one witness for each stated category.

    A SafeSchema category is legitimate only when it exists in cleaned history.
    After target-null filtering and bounded sampling, however, a rare legitimate
    category could disappear from the *actual* TabPFN fit frame.  Retaining one
    deterministic labeled witness prevents that accidental incompatibility
    without treating the query value as an unknown category.
    """
    witness_positions: set[int] = set()
    coverage_constraints: dict[str, int] = {}
    rng = np.random.default_rng(random_state)
    for requirement in requirements:
        scalar_kind = _categorical_scalar_kind(eligible[requirement.feature])
        values = _canonical_categorical_series(eligible[requirement.feature], scalar_kind)
        matches = np.flatnonzero(values.eq(requirement.canonical_value).to_numpy())
        if len(matches) == 0:
            raise CategoryUnavailableForTargetError(
                row_index=requirement.row_index,
                feature=requirement.feature,
                value=requirement.value,
            )
        witness_positions.add(int(rng.choice(matches)))
        coverage_constraints[requirement.feature] = (
            coverage_constraints.get(requirement.feature, 0) + 1
        )
    if len(witness_positions) > max_rows:
        raise CategoryCoverageCapacityError(
            "The number of stated categorical values exceeds the TabPFN context row limit."
        )
    if len(eligible) <= max_rows:
        return eligible, "full", coverage_constraints
    all_positions = np.arange(len(eligible))
    witness_array = np.fromiter(sorted(witness_positions), dtype=int)
    remaining = np.setdiff1d(all_positions, witness_array, assume_unique=True)
    fill_count = max_rows - len(witness_array)
    sampled_remaining = rng.choice(remaining, size=fill_count, replace=False)
    selected = np.concatenate((witness_array, sampled_remaining))
    rng.shuffle(selected)
    return (
        eligible.iloc[selected].copy(),
        f"random_{max_rows}_category_coverage",
        coverage_constraints,
    )


_NUMERIC_INFERRED_KINDS = frozenset(
    {
        "boolean",
        "decimal",
        "floating",
        "integer",
        "mixed-integer-float",
    }
)


def _categorical_scalar_kind(series: pd.Series) -> Literal["numeric", "string"]:
    """Infer a stable categorical scalar representation from history only."""
    non_null = series.dropna()
    if non_null.empty:
        # No query-independent scalar evidence exists. A string token contract
        # keeps the all-missing history column executable without interpreting
        # any query value or changing native missingness.
        return "string"
    inferred = pd.api.types.infer_dtype(non_null, skipna=True)
    return "numeric" if inferred in _NUMERIC_INFERRED_KINDS else "string"


def _canonical_categorical_value(
    value: object, scalar_kind: Literal["numeric", "string"]
) -> str | float:
    """Convert a validated scalar to the history-derived categorical representation."""
    if scalar_kind == "numeric":
        number = pd.to_numeric(pd.Series([value]), errors="raise").iloc[0]
        if pd.isna(number):
            raise TabPFNFrameContractError(
                "a non-null categorical query value cannot become missing during canonicalization"
            )
        return float(number)
    return str(value)


def _canonical_categorical_series(
    series: pd.Series, scalar_kind: Literal["numeric", "string"]
) -> pd.Series:
    if scalar_kind == "numeric":
        return pd.to_numeric(series, errors="coerce").astype("float64")
    return _string_categorical_series(series)


def _align_categorical_column(
    history: pd.Series,
    query: pd.Series,
    *,
    name: str,
    scalar_kind: Literal["numeric", "string"],
) -> tuple[pd.Series, pd.Series]:
    """Apply a history-derived scalar contract to fit and predict categoricals.

    TabPFN V2's ``fix_dtypes`` preserves an existing ``CategoricalDtype``.  A
    bare ``object`` query column is insufficient: for an all-missing sparse
    query Pandas would otherwise infer an empty/float category domain at
    predict time, while the fitted history has string or numeric categories.
    The two frames share both the scalar dtype and the closed vocabulary derived
    from history. Runtime validation has already required every non-null query
    category to match ``SafeSchema.allowed_values``; this function independently
    fails if a value would be lost while casting to the fitted history domain.
    """
    if scalar_kind == "numeric":
        history_values = _canonical_categorical_series(history, scalar_kind)
        if bool(history.notna().any() and history_values[history.notna()].isna().any()):
            raise ValueError(
                f"categorical history values for {name!r} do not share one numeric representation"
            )
        query_values = _canonical_categorical_series(query, scalar_kind)
        if bool(query.notna().any() and query_values[query.notna()].isna().any()):
            raise TabPFNFrameContractError(
                f"categorical query values for {name!r} are not representable as the history numeric scalar type"
            )
    else:
        history_values = _canonical_categorical_series(history, scalar_kind)
        query_values = _canonical_categorical_series(query, scalar_kind)
    history_categories = pd.unique(history_values.dropna())
    non_null_query = query_values[query_values.notna()]
    if bool((~non_null_query.isin(history_categories)).any()):
        raise TabPFNFrameContractError(
            f"categorical query values for {name!r} are outside the fitted history vocabulary"
        )
    shared_dtype = pd.CategoricalDtype(categories=history_categories)
    aligned_query = pd.Series(
        pd.Categorical(query_values, dtype=shared_dtype),
        index=query.index,
        name=query.name,
    )
    if bool(query_values.notna().any() and aligned_query[query_values.notna()].isna().any()):
        raise TabPFNFrameContractError(
            f"categorical query values for {name!r} are outside the fitted history vocabulary"
        )
    return (
        pd.Series(
            pd.Categorical(history_values, dtype=shared_dtype),
            index=history.index,
            name=history.name,
        ),
        aligned_query,
    )


def _string_categorical_series(series: pd.Series) -> pd.Series:
    """Preserve native missing values while giving all category tokens one scalar dtype."""
    return series.map(lambda value: np.nan if pd.isna(value) else str(value)).astype("object")


def assert_tabpfn_preprocessing_contract(context: ModelContext) -> None:
    """Revalidate an already prepared context immediately before prediction."""
    categorical = [context.feature_columns[index] for index in context.categorical_features_indices]
    _assert_tabpfn_preprocessing_contract(
        **_contract_view(
            context.x_history,
            context.x_query,
            context.feature_columns,
            categorical,
            list(context.datetime_columns) + list(context.text_columns),
        )
    )


def _assert_tabpfn_preprocessing_contract(
    *,
    x_history: pd.DataFrame,
    x_query: pd.DataFrame,
    categorical_features_indices: list[int],
) -> None:
    """Execute TabPFN's own CPU dtype path as a non-mutating compatibility check.

    This deliberately calls the exact 8.1.0 preprocessing functions that
    later run in ``fit``/``predict``.  It neither fits TabPFN nor uses the
    encoded arrays for prediction; it only rejects a frame construction that
    would otherwise make TabPFN warn and coerce query values at predict time.
    """
    from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
    from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder

    history_frame = fix_dtypes(x_history.copy(deep=True), categorical_features_indices)
    query_frame = fix_dtypes(x_query.copy(deep=True), categorical_features_indices)
    _assert_scalar_dtype_compatibility(history_frame, query_frame)
    encoder = get_ordinal_encoder()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r"Column\(s\) .* have a dtype at predict time that differs",
                category=UserWarning,
                module=r"tabpfn\.preprocessing\.clean",
            )
            process_text_na_dataframe(history_frame, ord_encoder=encoder, fit_encoder=True)
            process_text_na_dataframe(query_frame, ord_encoder=encoder, fit_encoder=False)
    except UserWarning as error:
        raise TabPFNFrameContractError("TabPFN fit/predict dtype contract would drift") from error


def _assert_scalar_dtype_compatibility(history: pd.DataFrame, query: pd.DataFrame) -> None:
    mismatched = [
        str(index)
        for index, column in enumerate(history.columns)
        if _dtype_kind(history[column].dtype) != _dtype_kind(query[column].dtype)
    ]
    if mismatched:
        raise TabPFNFrameContractError(
            f"TabPFN fit/predict scalar dtype mismatch at feature indices {', '.join(mismatched)}"
        )


def _dtype_kind(dtype: object) -> str:
    if isinstance(dtype, pd.CategoricalDtype):
        return dtype.categories.dtype.kind
    return getattr(dtype, "kind", "")
