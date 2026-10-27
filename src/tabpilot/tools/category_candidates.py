"""Deterministic, bounded candidate retrieval for deferred category domains."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from tabpilot.contracts import Scalar


def retrieve_category_candidates(
    submitted_value: Scalar,
    vocabulary: Iterable[Scalar],
    *,
    labeled_supported_values: Iterable[Scalar] = (),
    limit: int = 5,
) -> list[Scalar]:
    """Return lexical suggestions without changing a submitted category value.

    Lexical similarity is always the primary ordering.  Availability among rows
    with an observed hypothesized target is only a deterministic final tie-break;
    it cannot displace a semantically closer candidate merely because it is more
    convenient for the sampled context.
    """
    if limit <= 0:
        return []
    supported = {_normalise(value) for value in labeled_supported_values}
    query = _normalise(submitted_value)
    ranked = sorted(
        vocabulary,
        key=lambda value: _candidate_key(query, value, supported),
    )
    return ranked[:limit]


def _candidate_key(query: str, value: Scalar, supported: set[str]) -> tuple[object, ...]:
    candidate = _normalise(value)
    return (
        -_token_overlap(query, candidate),
        _normalised_edit_distance(query, candidate),
        abs(len(query) - len(candidate)),
        0 if candidate in supported else 1,
        candidate,
        str(value),
    )


def _normalise(value: Scalar) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()


def _token_overlap(left: str, right: str) -> int:
    return len(set(re.findall(r"\w+", left)) & set(re.findall(r"\w+", right)))


def _normalised_edit_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))
