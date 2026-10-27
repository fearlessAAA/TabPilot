"""Column-name canonicalization for model-supplied names."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Literal


def normalized_column_surface(value: str) -> str:
    """Normalize only presentation differences in a model-supplied column name.

    This intentionally does not recognize synonyms.  It is restricted to NFKC,
    case, trimming, and runs of whitespace/underscore/hyphen so a unique match
    remains a physical representation repair rather than semantic inference.
    """
    return re.sub(r"[\s_-]+", " ", unicodedata.normalize("NFKC", value).casefold().strip())


ColumnSurfaceStatus = Literal["exact", "canonicalized", "unknown", "ambiguous"]


def canonical_column_name(
    submitted: str,
    allowed_names: Iterable[str],
) -> tuple[str | None, ColumnSurfaceStatus]:
    """Return an exact name only when its surface-normalized match is unique."""
    names = list(allowed_names)
    if submitted in names:
        return submitted, "exact"
    surface = normalized_column_surface(submitted)
    matches = [name for name in names if normalized_column_surface(name) == surface]
    if len(matches) == 1:
        return matches[0], "canonicalized"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "unknown"
