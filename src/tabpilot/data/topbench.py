from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd

from tabpilot.contracts import RawTaskBundle
from tabpilot.data.cleaning import clean_frame


class UnsafeQueryPayloadError(ValueError):
    pass


def _read_query_payload(query_file: Path) -> tuple[str, str]:
    with query_file.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("query"), str) or not payload["query"].strip():
        raise UnsafeQueryPayloadError(f"{query_file} has no non-empty query")
    return str(payload.get("id", query_file.stem)), payload["query"]


def _load_history_and_current(
    task_dir: Path, *, load_current: bool
) -> tuple[pd.DataFrame, list[str], pd.DataFrame | None, str]:
    history_path = task_dir / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    # One-pass inference avoids chunk-dependent mixed dtypes and the associated
    # DtypeWarning. The cleaned history remains the sole source of the runtime
    # schema and TabPFN fit-time column contract.
    history, dropped_columns = clean_frame(pd.read_csv(history_path, low_memory=False))
    # ``current.csv`` also occurs in some non-candidate task directories.  A
    # provided cases table is bound only by the query-file data contract
    # (``*_current.json``), never by a task-family name or directory glob.
    current_path = task_dir / "current.csv"
    current = None
    if load_current and current_path.exists():
        # Use the same one-pass dtype inference as history.  This table is bound
        # only by its exact filename; test_current.csv and glob discovery are
        # intentionally unreachable from runtime.
        current, _ = clean_frame(pd.read_csv(current_path, low_memory=False))
    history_source_id = hashlib.sha256(str(history_path.resolve()).encode("utf-8")).hexdigest()[:24]
    return history, dropped_columns, current, history_source_id


def load_task_bundle(task_dir: Path, query_file: Path) -> RawTaskBundle:
    """Read only query text/id and CSVs; never deserialize ground_truth into runtime state."""
    task_dir, query_file = Path(task_dir), Path(query_file)
    query_id, query = _read_query_payload(query_file)
    is_current_query = query_file.stem.endswith("_current")
    history, dropped_columns, current, history_source_id = _load_history_and_current(
        task_dir, load_current=is_current_query
    )
    return RawTaskBundle(
        query_id=query_id,
        query=query,
        query_path=str(query_file),
        task_dir=str(task_dir),
        history=history,
        current=current,
        history_source_id=history_source_id,
        dropped_columns=dropped_columns,
    )
