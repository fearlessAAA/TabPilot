"""Reference query ``Q* = (y*, T*, X*)`` built from TopBench's annotations, for training only.

``y*``/``T*`` are the annotated target column and prediction type. ``X*`` holds the annotated
feature rows of the scenarios that must be sent to the TFM (``data/reference_query_rows.json``
lists them per request: every scenario for single-point prediction and decision making; for
treatment-effect analysis the changed scenario, plus the reference scenario only when the request
does not state its outcome). The annotation is read only by training code; nothing derived from it
reaches the agent's inputs.

PS-OPSD gives the teacher the raw specification (``privileged_specification``). SA-GRPO scores
against the canonical form (``build_reference_query``): each cell goes through
``prepare_tfm_input``'s own scalar validation, and a continuous number is replaced by the number
the request states for that field and scenario when that number is a rounding of the annotated
value (``3.39`` for an annotated ``3.387153393``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Context, Decimal, localcontext
from pathlib import Path
from typing import Any

import pandas as pd

from tabpilot.contracts import PredictionRow, PrepareTFMInputArguments, SafeSchema
from tabpilot.tools.canonicalization import canonical_column_name
from tabpilot.tools.prepare_tfm_input import _validate_request
from tabpilot.tools.run_store import RunStore
from tabpilot.training.structure_aware_reward import column_types, to_decimal

TREATMENT_EFFECT_TASK = "treatment_effect_analysis"
REFERENCE_SCENARIO, CHANGED_SCENARIO = "001", "002"
_MISSING_MARKERS = {"null", "none", "nan", ""}


class ReferenceQueryUnavailable(Exception):
    """The annotation cannot be expressed as a valid query; the request is left out of SA-GRPO."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass(frozen=True)
class ReferenceQuery:
    target_column: str
    prediction_type: str
    rows: dict[str, dict[str, Any]]

    def to_json(self) -> dict[str, object]:
        return {
            "target_column": self.target_column,
            "prediction_type": self.prediction_type,
            "rows": {sid: {k: _encode(v) for k, v in row.items()} for sid, row in self.rows.items()},
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ReferenceQuery:
        return cls(
            target_column=str(payload["target_column"]),
            prediction_type=str(payload["prediction_type"]),
            rows={sid: {k: _decode(v) for k, v in row.items()} for sid, row in payload["rows"].items()},
        )

    def prediction_rows(self) -> list[PredictionRow]:
        """``X*`` as a ``prepare_tfm_input`` payload (``row_id`` = scenario id)."""
        return [
            PredictionRow(row_id=sid, features={k: _json_scalar(v) for k, v in row.items()})
            for sid, row in self.rows.items()
        ]


def load_reference_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Per request: ``scenarios``, the scenarios to ``predict`` and, for treatment-effect analysis,
    whether the request states the reference outcome (``reference_status``)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


def privileged_specification(query_file: Path, predict: list[str]) -> dict[str, Any]:
    """The PS-OPSD teacher's privileged specification: target, type and the rows to predict."""
    ground_truth = json.loads(Path(query_file).read_text(encoding="utf-8")).get("ground_truth") or {}
    target = str(ground_truth.get("target_column") or "")
    prediction_type = ground_truth.get("task_sub_type")
    if not target or prediction_type not in {"classification", "regression"}:
        raise ValueError(f"{query_file}: annotation has no target or prediction type")
    scenarios = {str(item.get("scenario_id", "")): item.get("features") or {} for item in ground_truth.get("extracted_features") or []}
    if not set(predict) <= set(scenarios):
        raise ValueError(f"{query_file}: scenarios {sorted(predict)} are not all annotated")
    cases = [
        {
            "case_role": "predict",
            "source_scenario_id": sid,
            "features": {name: value for name, value in scenarios[sid].items() if name != target},
        }
        for sid in scenarios
        if sid in predict
    ]
    return {"target_column": target, "prediction_type": prediction_type, "cases": cases}


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_reference_query(
    *,
    query_file: Path,
    task: str,
    predict: list[str],
    schema: SafeSchema,
    history: pd.DataFrame,
    dropped_columns: list[str] | tuple[str, ...] = (),
) -> ReferenceQuery:
    """Canonical ``Q*``; raises ``ReferenceQueryUnavailable`` when it cannot be a valid query."""
    payload = json.loads(Path(query_file).read_text(encoding="utf-8"), parse_float=Decimal)
    ground_truth = payload.get("ground_truth") or {}
    target = str(ground_truth.get("target_column") or "")
    prediction_type = ground_truth.get("task_sub_type")
    if target not in schema.names:
        raise ReferenceQueryUnavailable("target_not_in_schema", repr(target))
    if prediction_type not in {"classification", "regression"}:
        raise ReferenceQueryUnavailable("invalid_prediction_type", repr(prediction_type))
    scenarios = {
        str(item.get("scenario_id", "")): dict(item.get("features") or {})
        for item in ground_truth.get("extracted_features") or []
    }
    store = RunStore(run_id="reference_query", query="", history=history, schema=schema, device="cpu")
    columns = column_types(schema, history)
    dropped = set(dropped_columns)
    canonical: dict[str, dict[str, Any]] = {}
    lexemes: dict[str, dict[str, str]] = {}
    for sid, features in sorted(scenarios.items()):
        kept = {name: value for name, value in features.items() if name != target and name not in dropped}
        cells, raw, failures = _canonical_row(kept, target, schema, store)
        if failures and sid in predict:
            raise ReferenceQueryUnavailable("unrepresentable_cell", f"scenario {sid}: {failures}")
        canonical[sid] = cells
        lexemes[sid] = {name: raw[name] for name in cells if columns[name].kind == "numeric"}

    treatment = task == TREATMENT_EFFECT_TASK
    anchors = _numeric_anchors(
        query=str(payload.get("query") or ""),
        column_names=list(schema.names),
        lexemes=lexemes,
        discrete=frozenset(name for name, column in columns.items() if column.discrete),
        roles={"reference": REFERENCE_SCENARIO, "modified": CHANGED_SCENARIO} if treatment else {},
    )
    rows: dict[str, dict[str, Any]] = {}
    for sid in sorted(predict):
        row = {
            name: _reference_number(anchors.get((sid, name)), lexemes[sid].get(name), value)
            if columns[name].kind == "numeric"
            else value
            for name, value in canonical.get(sid, {}).items()
        }
        if not row:
            raise ReferenceQueryUnavailable("empty_row", f"scenario {sid} has no usable cell")
        rows[sid] = row
    return ReferenceQuery(target_column=target, prediction_type=str(prediction_type), rows=rows)


def _canonical_row(
    features: dict[str, Any], target: str, schema: SafeSchema, store: RunStore
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Canonical cells through the tool's scalar validation, the numbers' original lexemes, and
    the cells the tool rejects."""
    submitted: dict[str, Any] = {}
    raw: dict[str, str] = {}
    for name, value in features.items():
        if value is None or (isinstance(value, str) and value.strip().casefold() in _MISSING_MARKERS):
            continue
        submitted[name] = float(value) if isinstance(value, Decimal) else value
        raw[name] = str(value)
    if not submitted:
        return {}, {}, []
    arguments = PrepareTFMInputArguments(
        target_column=target,
        prediction_type="regression",
        rows=[PredictionRow(row_id="reference", features=submitted)],
    )
    errors, _target, normalized, _warnings = _validate_request(arguments, store)
    failures = [f"{error.code}:{error.path}" for error in errors]
    cells = {name: value for name, value in (normalized[0].features if normalized else {}).items() if value is not None}
    lexemes: dict[str, str] = {}
    for name, lexeme in raw.items():
        canonical, _status = canonical_column_name(name, schema.names)
        if canonical in cells:
            lexemes[canonical] = lexeme
    return cells, {name: lexemes.get(name, str(cells[name])) for name in cells}, failures


# ---------------------------------------------------------------------------------------------
# Numeric anchoring: replace an annotated continuous number by the request's own rounded number.
#
# The request is split into clauses. A clause names a field when it mentions exactly one column
# (name tokens split on case/underscores, lightly stemmed) and contains exactly one number with
# no negation, bound or range. "from A to B" and "<change verb> ... to B" give role-attributed
# numbers for the reference and changed scenario. A request number q with last-digit quantum h
# anchors an annotated value g when q == g or q == round(g / h) * h (half-up or half-even);
# discrete columns only accept q == g.
# ---------------------------------------------------------------------------------------------

_DECIMAL_CONTEXT = Context(prec=200, Emin=-999999, Emax=999999)
_MULTIPLIERS = {
    "k": Decimal(1000), "thousand": Decimal(1000), "million": Decimal(10) ** 6, "mn": Decimal(10) ** 6,
    "billion": Decimal(10) ** 9, "bn": Decimal(10) ** 9,
}
_NUMBER_RE = re.compile(
    r"(?P<sign>[-+])?"
    r"(?P<body>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+|\.\d+)"
    r"(?:[eE](?P<exp>[-+]?\d+))?"
    r"(?P<unit>\s?%|\s?(?:k|thousand|million|mn|billion|bn)\b)?",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<!\d)\.(?!\d)|\.(?=\s)|\.$|[;!?\n]|,(?!\d{3}(?!\d))|\s+(?:and|but|while|whereas|plus)\s+",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(?:not|no|never|without|none|nor)\b|n't\b", re.IGNORECASE)
_BOUND_RE = re.compile(
    r"\b(?:at\s+least|at\s+most|up\s+to|more\s+than|less\s+than|fewer\s+than|greater\s+than|"
    r"higher\s+than|lower\s+than|over|under|above|below|between|exceed(?:s|ing)?|"
    r"or\s+more|or\s+less|by)\b",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(r"\d\s*(?:-|–|—|to)\s*[-+]?\d")
_FROM_TO_RE = re.compile(r"\bfrom\b(?P<a>.*?)\bto\b(?P<b>.*)$", re.IGNORECASE)
_CHANGE_TO_RE = re.compile(
    r"\b(?:change[sd]?|changing|increase[sd]?|increasing|raise[sd]?|raising|lower(?:s|ed|ing)?|"
    r"reduce[sd]?|reducing|decrease[sd]?|decreasing|set|sets|setting|drop(?:s|ped|ping)?|"
    r"bump(?:s|ed|ing)?|move[sd]?|moving|boost(?:s|ed|ing)?|cut(?:s|ting)?|switch(?:es|ed|ing)?|"
    r"upgrade[sd]?|upgrading|downgrade[sd]?|adjust(?:s|ed|ing)?|went|go(?:es)?|going|rose|rise[sn]?|"
    r"fell|fall(?:s|ing)?|grow(?:s|n|ing)?|grew|become[s]?|became|jump(?:s|ed|ing)?)\b"
    r"(?P<middle>.*?)\bto\b(?P<b>.*)$",
    re.IGNORECASE,
)
_STOP_TOKENS = frozenset({"n", "no", "num", "of", "the", "a", "an", "is", "id", "per", "in", "at", "to", "and", "or"})
_SUFFIXES = ("iness", "ness", "ities", "ity", "ies", "ing", "ed", "es", "s", "y")


@dataclass(frozen=True)
class _Number:
    value: Decimal
    quantum: Decimal
    unit: str


@dataclass(frozen=True)
class _Mention:
    field: str
    number: _Number
    role: str | None


def _numeric_anchors(
    *,
    query: str,
    column_names: list[str],
    lexemes: dict[str, dict[str, str]],
    discrete: frozenset[str],
    roles: dict[str, str],
) -> dict[tuple[str, str], Decimal]:
    """The anchored reference value per (scenario, feature); an unanchored cell keeps its
    annotated value."""
    mentions: dict[str, list[_Mention]] = {}
    for mention in _field_mentions(query, column_names):
        mentions.setdefault(mention.field, []).append(mention)
    anchors: dict[tuple[str, str], Decimal] = {}
    for feature in sorted({name for cells in lexemes.values() for name in cells}):
        scenarios = [sid for sid in sorted(lexemes) if feature in lexemes[sid]]
        gold = {sid: to_decimal(lexemes[sid][feature]) for sid in scenarios}
        if any(value is None for value in gold.values()):
            continue
        exact_only = feature in discrete
        assigned: dict[str, _Number] = {}
        failed: set[str] = set()
        for mention in mentions.get(feature, []):
            sid = roles.get(mention.role or "")
            if sid is None or sid not in gold:
                continue
            if sid in assigned and assigned[sid].value != mention.number.value:
                failed.add(sid)
            else:
                assigned[sid] = mention.number
        role_values = {number.value for number in assigned.values()}
        unattributed: list[_Number] = []
        for mention in mentions.get(feature, []):
            value = mention.number.value
            if mention.role is None and value not in role_values and all(n.value != value for n in unattributed):
                unattributed.append(mention.number)
        remaining = [sid for sid in scenarios if sid not in assigned and sid not in failed]
        if remaining and unattributed:
            if len(scenarios) == 1 or len({gold[sid] for sid in remaining}) == 1:
                if len(unattributed) == 1:
                    assigned.update({sid: unattributed[0] for sid in remaining})
            else:
                # Distinct annotated values in scenario order pair with request numbers in text order.
                distinct = list(dict.fromkeys(gold[sid] for sid in remaining))
                if len(unattributed) == len(distinct) and all(
                    _compatible(value, unattributed[index], exact_only) for index, value in enumerate(distinct)
                ):
                    assigned.update({sid: unattributed[distinct.index(gold[sid])] for sid in remaining})
        for sid, number in assigned.items():
            if sid not in failed and _compatible(gold[sid], number, exact_only):
                anchors[(sid, feature)] = number.value
    return anchors


def _compatible(gold: Decimal, number: _Number, exact_only: bool) -> bool:
    with localcontext(_DECIMAL_CONTEXT):
        if number.value == gold:
            return True
        if exact_only or number.quantum <= 0:
            return False
        return any(
            (gold / number.quantum).to_integral_value(rounding=mode) * number.quantum == number.value
            for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN)
        )


def _field_mentions(query: str, column_names: list[str]) -> list[_Mention]:
    text = unicodedata.normalize("NFKC", query).replace("−", "-")
    phrases = {name: _field_phrases(name) for name in column_names}
    mentions: list[_Mention] = []
    for clause in (part.strip() for part in _CLAUSE_SPLIT_RE.split(text) if part and part.strip()):
        words = [_stem(word) for word in re.findall(r"[a-z]+", clause.casefold())]
        fields = _mentioned_fields(words, phrases)
        if len(fields) != 1 or _NEGATION_RE.search(clause):
            continue
        field = next(iter(fields))
        roles = _role_numbers(clause)
        if roles is not None:
            mentions.extend(_Mention(field, number, role) for role, number in roles)
            continue
        if _BOUND_RE.search(clause) or _RANGE_RE.search(clause):
            continue
        numbers = _numbers(clause)
        if len(numbers) == 1:
            mentions.append(_Mention(field, numbers[0], None))
    return mentions


def _role_numbers(clause: str) -> list[tuple[str, _Number]] | None:
    match = _FROM_TO_RE.search(clause)
    if match:
        before, after = _numbers(match.group("a")), _numbers(match.group("b"))
        exact = not any(_BOUND_RE.search(side) or _RANGE_RE.search(side) for side in (match.group("a"), match.group("b")))
        if exact and len(before) == 1 and len(after) == 1:
            return [("reference", before[0]), ("modified", after[0])]
        return None
    match = _CHANGE_TO_RE.search(clause)
    if match:
        after = _numbers(match.group("b"))
        tail = match.group("b")
        if not _numbers(match.group("middle")) and len(after) == 1 and not _BOUND_RE.search(tail) and not _RANGE_RE.search(tail):
            return [("modified", after[0])]
    return None


def _numbers(text: str) -> list[_Number]:
    numbers: list[_Number] = []
    for match in _NUMBER_RE.finditer(text):
        sign = match.group("sign") or ""
        start = match.start("sign") if sign else match.start("body")
        if sign and start > 0 and not (text[start - 1].isspace() or text[start - 1] in "([:=~$£€"):
            sign, start = "", match.start("body")  # "3-5" or "2014-02": a separator, not a sign
        if start > 0 and (text[start - 1].isalpha() or text[start - 1] in "_."):
            continue  # part of an identifier-like token
        digits = match.group("body").replace(",", "")
        decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
        unit = (match.group("unit") or "").strip().casefold()
        with localcontext(_DECIMAL_CONTEXT):
            value = Decimal(sign + digits)
            quantum = Decimal(1).scaleb(-decimals)
            if match.group("exp") is not None:
                shift = int(match.group("exp"))
                value, quantum = value.scaleb(shift), quantum.scaleb(shift)
            if unit and unit != "%":
                value, quantum = value * _MULTIPLIERS[unit], quantum * _MULTIPLIERS[unit]
        numbers.append(_Number(value=value, quantum=quantum, unit=unit))
    return numbers


def _field_phrases(column_name: str) -> list[tuple[str, ...]]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", column_name)
    spaced = re.sub(r"([A-Za-z])(\d)", r"\1 \2", spaced)
    tokens = [token for token in re.split(r"[^A-Za-z]+", spaced.casefold()) if token]
    if not tokens:
        return []
    phrases = [tuple(_stem(token) for token in tokens)]
    content = [token for token in tokens if len(token) > 1 and token not in _STOP_TOKENS]
    if len(tokens) > 1 and content and content != tokens:
        phrases.append(tuple(_stem(token) for token in content))
    return phrases


def _mentioned_fields(words: list[str], phrases: dict[str, list[tuple[str, ...]]]) -> set[str]:
    """Columns mentioned in ``words``, dropping a mention nested in a longer column's mention."""
    spans = [
        (start, start + len(phrase), name)
        for name, options in phrases.items()
        for phrase in options
        if phrase and not (len(phrase) == 1 and len(phrase[0]) <= 1)
        for start in range(len(words) - len(phrase) + 1)
        if tuple(words[start:start + len(phrase)]) == phrase
    ]
    return {
        name
        for start, end, name in spans
        if not any(
            other != name and other_start <= start and end <= other_end and other_end - other_start > end - start
            for other_start, other_end, other in spans
        )
    }


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _reference_number(anchor: Decimal | None, lexeme: str | None, value: Any) -> Any:
    """The anchored request number, else the annotated number at its full written precision."""
    if anchor is not None:
        return anchor
    exact = to_decimal(lexeme)
    return value if exact is None else exact


def _encode(value: Any) -> Any:
    return {"decimal": str(value)} if isinstance(value, Decimal) else value


def _decode(value: Any) -> Any:
    return Decimal(value["decimal"]) if isinstance(value, dict) and set(value) == {"decimal"} else value


def _json_scalar(value: Any) -> Any:
    """The JSON scalar an agent would submit for a canonical value."""
    if isinstance(value, Decimal):
        text = str(value)
        return int(value) if value == value.to_integral_value() and "." not in text and "E" not in text.upper() else float(value)
    return value
