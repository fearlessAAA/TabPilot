"""Small, cached history summaries used by inspection only.

These profiles contain no query answer or benchmark task metadata and are never
used as a runtime acceptance gate for agent-proposed numeric values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NumericColumnProfile:
    non_null_count: int
    minimum: float | None
    q05: float | None
    median: float | None
    q95: float | None
    maximum: float | None


def build_numeric_column_profile(series: pd.Series) -> NumericColumnProfile:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return NumericColumnProfile(0, None, None, None, None, None)
    return NumericColumnProfile(
        non_null_count=int(len(numeric)),
        minimum=float(numeric.min()),
        q05=float(numeric.quantile(0.05)),
        median=float(numeric.median()),
        q95=float(numeric.quantile(0.95)),
        maximum=float(numeric.max()),
    )
