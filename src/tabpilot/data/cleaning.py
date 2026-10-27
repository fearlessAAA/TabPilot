from __future__ import annotations

import pandas as pd


def clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop only universally empty and generated-index columns, preserving real CSV-only fields."""
    dropped = [
        column
        for column in frame.columns
        if str(column).startswith("Unnamed:") or frame[column].isna().all()
    ]
    return frame.drop(columns=dropped), dropped
