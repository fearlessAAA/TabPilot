"""Deterministic full-table prediction materialization for provided cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tabpilot.contracts import PredictionArtifact


CSV_FLOAT_FORMAT = "%.17g"


def materialize_current_table_artifacts(
    *,
    output_dir: Path,
    history_columns: list[str],
    target_column: str,
    current: pd.DataFrame,
    prediction: PredictionArtifact,
) -> dict[str, Any]:
    """Write all runtime-bound cases in history column order.

    Evaluation applies any task-specific filter, sort, and cutoff after runtime.
    The agent therefore never discards candidate rows before the scorer sees them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    by_row_id = {record.row_id: record for record in prediction.records}
    full = current.copy(deep=True)
    target_values = [_target_value(by_row_id[f"case_{ordinal + 1}"]) for ordinal in range(len(full))]
    full[target_column] = target_values
    ordered = _history_ordered_frame(full, history_columns)
    prediction_path = output_dir / "prediction_table.csv"
    _write_csv(ordered, prediction_path)

    result = ordered.copy()
    result_path = output_dir / "result.csv"
    _write_csv(result, result_path)
    return {
        "prediction_table_path": str(prediction_path),
        "result_path": str(result_path),
        "prediction_row_ids": [record.row_id for record in prediction.records],
        "result_row_ids": [record.row_id for record in prediction.records],
        "csv_float_format": CSV_FLOAT_FORMAT,
        "newline": "LF",
    }


def _history_ordered_frame(frame: pd.DataFrame, history_columns: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in history_columns:
        result[column] = frame[column] if column in frame.columns else pd.NA
    return result


def _target_value(record: Any) -> object:
    return record.predicted_value if record.predicted_value is not None else record.predicted_label


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if bool(frame.isin([float("inf"), float("-inf")]).any().any()):
        raise ValueError("cannot materialize non-finite values")
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        na_rep="",
        float_format=CSV_FLOAT_FORMAT,
    )
