"""Structure-Aware Reward of SA-GRPO: ``R = V * I[y = y*] * I[T = T*] * S(X, X*)``.

``V`` is execution validity, ``y``/``T`` the target and prediction type of the latest valid
``prepare_tfm_input`` call, and ``S`` the scenario similarity between its canonical rows ``X`` and
the reference rows ``X*``: per-row cell F1, summed over an optimal one-to-one row matching
(Hungarian algorithm) and divided by ``max(|X|, |X*|, 1)``.

Cells are compared by the column's effective kind under the tool's native feature handling:
numbers as exact decimals (continuous columns also within 4 ulp of the reference), categoricals
by their canonical value, dates as UTC timestamps, free text and string identifiers exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import pandas as pd

from tabpilot.contracts import ColumnKind, SafeSchema
from tabpilot.tfm.context import native_datetime_columns, to_utc_datetime
from tabpilot.tools.canonicalization import canonical_column_name

CellKind = Literal["numeric", "categorical", "text", "datetime"]


@dataclass(frozen=True)
class ColumnType:
    kind: CellKind
    # Identifiers and integer-valued numeric columns only match exactly.
    discrete: bool


def column_types(schema: SafeSchema, history: pd.DataFrame) -> dict[str, ColumnType]:
    """The scalar kind each column's cells take after ``prepare_tfm_input`` canonicalization."""
    dates = native_datetime_columns(history, schema)
    types: dict[str, ColumnType] = {}
    for spec in schema.history_columns:
        kind = spec.kind
        if spec.is_identifier and kind == ColumnKind.CATEGORICAL:
            numeric = spec.name in history.columns and pd.api.types.is_numeric_dtype(history[spec.name])
            kind = ColumnKind.NUMERIC if numeric else ColumnKind.OPEN_TEXT
        if kind == ColumnKind.NUMERIC:
            cell_kind: CellKind = "numeric"
        elif kind == ColumnKind.CATEGORICAL:
            cell_kind = "categorical"
        elif spec.name in dates:
            cell_kind = "datetime"
        else:
            cell_kind = "text"
        discrete = bool(spec.is_identifier) or (cell_kind == "numeric" and _integer_valued(history, spec.name))
        types[spec.name] = ColumnType(kind=cell_kind, discrete=discrete)
    return types


def to_decimal(value: object) -> Decimal | None:
    """Exact decimal of a JSON number, decimal string or float (a float via its shortest repr)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, int):
            result = Decimal(value)
        elif isinstance(value, float):
            result = Decimal(repr(value))
        elif isinstance(value, str) and value.strip():
            result = Decimal(value.strip().replace("−", "-"))
        else:
            return None
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def cells_equal(column: ColumnType, value: Any, reference: Any) -> bool:
    if value is None or reference is None:
        return False
    if column.kind == "numeric":
        return _numbers_equal(value, reference, exact=column.discrete)
    if column.kind == "categorical":
        if isinstance(value, bool) != isinstance(reference, bool):
            return False
        if isinstance(value, (int, float)) and isinstance(reference, (int, float)) and not isinstance(value, bool):
            return math.isfinite(float(value)) and float(value) == float(reference)
        return type(value) is type(reference) and value == reference
    if column.kind == "datetime":
        parsed = to_utc_datetime(pd.Series([str(value), str(reference)]))
        if parsed.notna().all():
            return bool(parsed.iloc[0] == parsed.iloc[1])
    return str(value) == str(reference)


def row_similarity(row: dict[str, Any], reference: dict[str, Any], columns: dict[str, ColumnType]) -> float:
    """Cell F1 ``2|x ∩ x*| / (|x| + |x*|)`` over non-missing cells; 0 when both rows are empty."""
    cells = {name: value for name, value in row.items() if value is not None}
    reference_cells = {name: value for name, value in reference.items() if value is not None}
    matched = sum(
        1
        for name, value in cells.items()
        if name in reference_cells and name in columns and cells_equal(columns[name], value, reference_cells[name])
    )
    total = len(cells) + len(reference_cells)
    return 2 * matched / total if total else 0.0


def scenario_similarity(
    rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], columns: dict[str, ColumnType]
) -> float:
    """``S``: the best one-to-one row matching's total similarity over ``max(|X|, |X*|, 1)``."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not rows or not reference_rows:
        return 0.0
    weights = np.array([[row_similarity(row, reference, columns) for row in rows] for reference in reference_rows])
    matched_rows, matched_columns = linear_sum_assignment(weights, maximize=True)
    return float(weights[matched_rows, matched_columns].sum()) / max(len(rows), len(reference_rows), 1)


@dataclass(frozen=True)
class RewardBreakdown:
    validity: int
    target_correct: int
    type_correct: int
    similarity: float | None
    reward: float

    def to_dict(self) -> dict[str, object]:
        return {
            "V": self.validity,
            "target_correct": self.target_correct,
            "type_correct": self.type_correct,
            "S": self.similarity,
            "R": self.reward,
        }


def structure_aware_reward(
    *,
    valid: bool,
    target_column: str | None,
    prediction_type: str | None,
    rows: list[dict[str, Any]] | None,
    reference_target: str,
    reference_type: str,
    reference_rows: list[dict[str, Any]],
    schema_names: list[str],
    columns: dict[str, ColumnType],
) -> RewardBreakdown:
    """``target_column``/``prediction_type``/``rows`` come from the latest valid prepare call."""
    target_correct = 0
    if target_column is not None:
        canonical, _status = canonical_column_name(target_column, schema_names)
        target_correct = int(canonical == reference_target)
    type_correct = int(prediction_type == reference_type)
    similarity = scenario_similarity(rows, reference_rows, columns) if valid and rows is not None else None
    reward = float(valid) * target_correct * type_correct * (similarity or 0.0)
    return RewardBreakdown(
        validity=int(valid),
        target_correct=target_correct,
        type_correct=type_correct,
        similarity=similarity,
        reward=reward,
    )


def _numbers_equal(value: Any, reference: Any, *, exact: bool) -> bool:
    if isinstance(value, bool):
        return False
    left, right = to_decimal(value), to_decimal(reference)
    if left is None or right is None:
        return False
    if left == right:
        return True
    if exact:
        return False
    try:
        left_float, right_float = float(left), float(right)
    except (OverflowError, ValueError):
        return False
    if not (math.isfinite(left_float) and math.isfinite(right_float)):
        return False
    return abs(left_float - right_float) <= 4 * math.ulp(right_float)


def _integer_valued(history: pd.DataFrame, name: str) -> bool:
    if name not in history.columns:
        return False
    values = pd.to_numeric(history[name], errors="coerce").dropna()
    return not values.empty and bool(((values - values.round()).abs() == 0).all())
