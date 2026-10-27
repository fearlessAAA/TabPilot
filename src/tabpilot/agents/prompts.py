"""Every string the TabPilot agents see: system prompts, tool descriptions and feedback.

These texts are part of the trained checkpoints; keep them byte-for-byte stable.
"""

from __future__ import annotations

import json
from typing import Any

# --- Execution Agent (Predictive Reasoning path) -------------------------------------------------
REFERENCE_OUTCOME_POLICY = "Use an explicitly stated reference outcome when one exists; otherwise compare returned row predictions."

PREDICTIVE_CAUSALITY_POLICY = "State causal limitations when relevant, but do not refuse a requested predictive comparison."

EXECUTION_PROMPT_HEADER = """You are the Execution Agent for a tabular prediction system. Work on one tabular prediction request using
the oracle-free SafeSchema and an optional advisory target hint. The schema contains no evaluation answer or gold label."""

EXECUTION_DECIDE_PROMPT = """DECIDE
- Infer the target from the requested outcome and decide type from outcome meaning: Measured amounts with
  meaningful arithmetic differences and averages imply regression; Labels, states, encoded categories, and ordinal bands imply
  classification even when stored as numbers. Storage dtype and cardinality alone do not decide meaning.
- If a planner target hint is supplied, use it only as a hypothesis about the requested outcome. The original request
  remains the only source for case facts and sparse rows; never copy, omit, or infer a feature from the target hint.
- If SafeSchema reports provided_cases, that runtime-bound table is the complete prediction set. Its values are
  unavailable to you and predicted automatically: submit no rows."""

EXECUTION_BUILD_PROMPT = f"""BUILD
- Every target and feature key must copy a SafeSchema name character-for-character, including spaces and case.
  Never rename, translate, or snake_case a column name.
- A row is exactly {{"row_id":"case_1","features":{{"Feature Name":"value"}}}}. Nest every feature inside
  features. Put only user-stated non-target facts there; unstated facts stay native missing.
- For multiple independent objects, make one row per object. Copy every stated value only into the row for the
  object it describes; never transfer, merge, swap, or paraphrase values across objects. Account for every factual
  statement that describes each model-authored case. Account for every factual statement before preparation.
- Omit every feature the user did not state.
- Build rows only for outcomes that need prediction. A reference outcome explicitly stated in the request remains
  a natural-language fact for the final answer; do not predict it again merely for comparison. If a reference is
  not stated, build every row necessary to answer.
- {REFERENCE_OUTCOME_POLICY}
- {PREDICTIVE_CAUSALITY_POLICY}
- When a request modifies one stated state, inherit its stated unchanged facts to the modified row and replace only
  changed facts. Never transfer facts between independent objects. Never put a target or identifier in features.
- For deferred categories, submit the user's wording. Candidate repairs are hints, never automatic corrections.

EXAMPLES (fictional; copy the exact SafeSchema names in a real call)
- Request: "Unit A has Load Weight 12.1 and Humidity 0.4. Unit B has Load Weight 9.8 and Humidity 0.7. Which
  unit is more likely to fail?"  Build two rows: unit_a has only Load Weight=12.1 and Humidity=0.4; unit_b has
  only Load Weight=9.8 and Humidity=0.7. Do not move 12.1 or 0.4 into unit_b.
- Request: "Candidate North has Annual Income 62000 and Credit Score 710. Candidate South has Annual Income
  48000 and Credit Score 660. Which candidate should be selected?"  Build two rows, preserving each candidate's
  own two values. Both requested outcomes are unknown, so neither belongs in known information."""

EXECUTION_ACT_PROMPT = """ACT
- Make exactly one tool call per turn. inspect_history is optional. prepare_tfm_input validates the physical
  boundary and returns a prepared_id; tfm_predict requires a prepared_id returned by prepare_tfm_input. Repair only reported paths and do
  not resend unchanged arguments. An error naming a feature path is repaired by fixing that feature, never by
  changing the target away from the requested outcome. Warnings do not require rewriting a valid row.
"""

# Every column type, identifiers included, is a TFM feature, so a stated identifier is submitted.
EXECUTION_BUILD_PROMPT = EXECUTION_BUILD_PROMPT.replace(
    "Never put a target or identifier in features.",
    "Never put a target in features. Include an identifier feature only when its value is stated "
    "for that case; copy its SafeSchema column name exactly.",
)

EXECUTION_SYSTEM_PROMPT = f"""{EXECUTION_PROMPT_HEADER}

{EXECUTION_DECIDE_PROMPT}

{EXECUTION_BUILD_PROMPT}

{EXECUTION_ACT_PROMPT}"""

# --- Execution Agent (Ranking and Filtering path) -------------------------------------------------
RANKING_SYSTEM_PROMPT = """You are a provided-table semantic agent for tabular prediction.
A runtime-bound table already contains every case that must be predicted. Decide only the requested
target column and whether that outcome is classification or regression. Do not answer the request.

DECIDE
- Copy target_column exactly from <safe_schema>, preserving every space, punctuation mark, and capital letter.
- Choose regression only for open-ended measured amounts whose arithmetic differences and averages are meaningful.
- Choose classification for labels, states, encoded categories, numeric grades, ratings, and ordinal bands, even when stored
  as numbers. Storage dtype and cardinality alone do not determine the meaning.

BOUNDARY
- The supplied cases are already complete and authoritative. Do not enumerate, reconstruct, reorder, or sample them.
  Their row values are unavailable to you.
- Do not infer case facts from historical examples. Historical schema describes the training population only.
- Submit exactly one target/type decision through submit_provided_table_intent. Do not write a final answer.

EXAMPLE A — fictional classification
Request: "For every supplied inspection record, predict its Safety Grade."
SafeSchema contains "Safety Grade" with numeric grade values.
Submit target_column="Safety Grade", prediction_type="classification".

EXAMPLE B — fictional regression
Request: "Estimate Delivery Time for every supplied shipment."
SafeSchema contains "Delivery Time" measured in hours.
Submit target_column="Delivery Time", prediction_type="regression".
"""

# --- Tool descriptions ----------------------------------------------------------------------------
INSPECT_HISTORY_TOOL_DESCRIPTION = (
    "Inspect a bounded view of cleaned historical training data only when a concrete uncertainty blocks the next "
    "prediction action. profiles gives per-column type, scale, range, distinct-value, or encoding evidence; rows "
    "gives at most three joint examples for co-occurrence. Omit columns to inspect every SafeSchema column in profiles "
    "or eligible columns in rows. Observations describe the training population, never unstated "
    "facts about the requested case."
)
PREPARE_TFM_INPUT_TOOL_DESCRIPTION = (
    "Every target and feature key must exactly copy a SafeSchema column name, including spaces, punctuation, and "
    "capitalization; never rename or snake_case it. Physically validate target, prediction type, and sparse rows, "
    "then construct a model-ready context. Every feature must be nested in a row's features object: "
    'the row shape is: {"row_id": "case_1", "features": {"Feature Name": "value"}}. If SafeSchema reports provided_cases, omit '
    "rows because the runtime predicts every supplied case. This tool never infers semantics or corrects values."
)
SUBMIT_PROVIDED_TABLE_INTENT_TOOL_DESCRIPTION = (
    "Submit the exact SafeSchema target column and its semantic prediction type for the already-bound candidate "
    "table. The table itself is immutable and is prepared and predicted deterministically after this submission. "
    "This tool does not accept rows, filters, ranking rules, or prediction values."
)
TFM_PREDICT_TOOL_DESCRIPTION = (
    "Run the tabular foundation model for a prepared_id returned by successful prepare_tfm_input in this run. Return predictions "
    "keyed by row_id: original-space labels and probabilities for classification, values for regression."
)
FEATURES_FIELD_DESCRIPTION = "Sparse user-stated non-target facts using exact SafeSchema column names."
ROW_ID_FIELD_DESCRIPTION = "A stable identifier for one unknown case that requires prediction."
ROWS_FIELD_DESCRIPTION = "Sparse rows for unknown outcomes. Submit an empty list when SafeSchema reports provided_cases."
TARGET_COLUMN_FIELD_DESCRIPTION = "Exact SafeSchema history column name for the requested outcome."
PREDICTION_TYPE_FIELD_DESCRIPTION = "classification for labels/states/bands; regression for measured quantities."
PROVIDED_TARGET_COLUMN_FIELD_DESCRIPTION = (
    "Exact SafeSchema history column holding the requested outcome; preserve spaces, punctuation, and case."
)
PROVIDED_PREDICTION_TYPE_FIELD_DESCRIPTION = (
    "classification for labels, states, grades, ratings, or encoded categories; regression for open-ended measured amounts."
)
INSPECT_VIEW_FIELD_DESCRIPTION = "profiles for marginal representation evidence; rows for a few joint examples."
INSPECT_COLUMNS_FIELD_DESCRIPTION = "Exact SafeSchema columns; omit to inspect all eligible columns for the selected view."
INSPECT_MAX_EXAMPLES_FIELD_DESCRIPTION = "Number of representative values or rows from 0 through 3."
PREPARED_ID_FIELD_DESCRIPTION = "prepared_id returned by successful prepare_tfm_input in this run."

# --- Response Agent -------------------------------------------------------------------------------
RESPONSE_SYSTEM_PROMPT = (
    "You are answering a question about tabular data. "
    "Use only the locked prediction facts provided to you; never invent a prediction value."
)
RESPONSE_REQUEST = "Answer the original request using the locked prediction facts."

# --- Protocol feedback and tool messages ---------------------------------------------------------
EMPTY_RESPONSE_FEEDBACK = "Your reply had no visible answer. Return a non-empty concise natural-language final answer using the successful prediction; do not call a tool."
RESPONSE_TOOL_CALL_FEEDBACK = "Tools are unavailable after prediction. Return concise final answer text using existing predictions."
PREMATURE_FINAL_ANSWER_FEEDBACK = "No successful model prediction exists yet. Make one tool call that advances preparation or prediction."
MULTIPLE_TOOL_CALLS_FEEDBACK = "You issued multiple tool calls; none ran. Choose one useful next action."
REPEATED_INVALID_ACTION_FEEDBACK = "The prior invalid attempt made no progress. Change the reported field paths before resubmitting."
REPEATED_SUCCESSFUL_ACTION_FEEDBACK = "This exact request has already succeeded and will not provide new information when repeated. Choose a different tool or materially change the request arguments."
INSPECTION_POPULATION_NOTE = "These observations describe historical training data, not unstated facts about the current case."
TFM_DTYPE_CONTRACT_FEEDBACK = "The history/query frames violate the model's dtype contract. This is a non-retryable system error; do not change semantic facts or fill missing features."
TFM_CAPABILITY_FEEDBACK = "The configured tabular foundation model cannot predict this target (for example, it has more classes than the model supports). This is a non-retryable system limit; do not change the target or prediction type to work around it."
UNSUPPORTED_CATEGORY_FEEDBACK = "This categorical value is outside SafeSchema's closed vocabulary. Use a supported value only when the request supports it; otherwise omit the feature."
AMBIGUOUS_CATEGORY_FEEDBACK = "This categorical value matches more than one allowed value after normalization. Use one exact value only when the request determines it; otherwise omit the feature."
DEFERRED_CATEGORY_REPAIR_FEEDBACK = "This value is outside the closed historical vocabulary. Candidates are hints, not automatic corrections; choose one only when the request supports it."
CATEGORY_WITHOUT_LABELED_HISTORY_FEEDBACK = "This allowed category has no labeled history support for the hypothesized target. Omit it rather than substituting another category."
PROVIDED_CASES_IGNORED_WARNING = "A task-provided cases table is the complete prediction set. Omit rows entirely; every supplied case is predicted automatically."
ROWS_REQUIRED_FEEDBACK = "No task-provided cases table exists, so submit at least one sparse row for an outcome that needs prediction."
PROVIDED_TARGET_MISSING_FEEDBACK = "The provided cases table cannot contain populated target values."
PROVIDED_INTENT_REPAIR_FEEDBACK = (
    "Fix only the reported target_column or prediction_type issue and resubmit the same intent. "
    "The supplied table is already bound; do not describe rows or answer the request."
)


def feature_excluded_from_model_message(feature: str, kind: str) -> str:
    return f"{feature!r} has kind {kind!r} and is excluded from the model feature matrix."


def misnested_features_message(row_index: int, columns: list[str]) -> str:
    return f"Move feature-column keys {', '.join(columns)} into rows[{row_index}].features. Preserve their values; do not delete them."


def execution_messages(query: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Initial Execution Agent messages on the Predictive Reasoning path."""
    return _messages(EXECUTION_SYSTEM_PROMPT, query, schema)


def ranking_messages(query: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Execution Agent messages on the Ranking and Filtering path."""
    return _messages(RANKING_SYSTEM_PROMPT, query, schema)


def _messages(system: str, query: str, schema: dict[str, Any]) -> list[dict[str, Any]]:
    content = f"<user_query>\n{query}\n</user_query>\n\n<safe_schema>\n{json.dumps(schema, ensure_ascii=False)}\n</safe_schema>"
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def response_messages(*, query: str, case_prediction_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Response Agent input: the request and the locked TFM predictions only."""
    content = (
        f"<original_request>\n{query}\n</original_request>\n\n"
        f"<locked_prediction_facts>\n{_render_locked_case_facts(case_prediction_facts)}\n</locked_prediction_facts>"
    )
    content += f"\n\n{RESPONSE_REQUEST}"
    return [{"role": "system", "content": RESPONSE_SYSTEM_PROMPT}, {"role": "user", "content": content}]


def _render_locked_case_facts(case_prediction_facts: list[dict[str, Any]]) -> str:
    """Render locked facts as self-contained lines, never JSON to be mentally joined."""
    if not case_prediction_facts:
        return "No locked prediction facts were supplied."
    return "\n".join(_render_locked_case_fact(fact) for fact in case_prediction_facts)


def _render_locked_case_fact(fact: dict[str, Any]) -> str:
    if fact.get("provided_cases") is True:
        return _render_provided_case_summary(fact)

    row_id = _render_fact_scalar(fact.get("row_id"))
    target = _render_fact_scalar(fact.get("target_column"))
    label = fact.get("predicted_label")
    value = fact.get("predicted_value")
    if label is not None:
        prediction = f'predicted {target} = {_render_fact_scalar(label)}'
    else:
        prediction = f'predicted {target} = {_render_fact_scalar(value)}'
    probabilities = fact.get("probabilities")
    probability_text = ""
    if isinstance(probabilities, dict) and probabilities:
        rendered = ", ".join(
            f"{_render_fact_scalar(key)}={_render_fact_scalar(probabilities[key])}"
            for key in sorted(probabilities, key=str)
        )
        probability_text = f"; probabilities: {rendered}"
    features = fact.get("stated_features")
    feature_text = ""
    if isinstance(features, dict) and features:
        rendered_features = ", ".join(
            f"{key}={_render_fact_scalar(features[key])}" for key in sorted(features)
        )
        feature_text = f"; request-stated features: {rendered_features}"
    return f"Case {row_id}{feature_text} -> {prediction}{probability_text}."


def _render_provided_case_summary(fact: dict[str, Any]) -> str:
    count = _render_fact_scalar(fact.get("provided_row_count"))
    target = _render_fact_scalar(fact.get("target_column"))
    prediction_type = _render_fact_scalar(fact.get("prediction_type"))
    labels = fact.get("predicted_label_counts")
    if isinstance(labels, dict) and labels:
        rendered = ", ".join(
            f"{_render_fact_scalar(key)}={_render_fact_scalar(labels[key])}" for key in sorted(labels, key=str)
        )
        summary = f"predicted label counts: {rendered}"
    else:
        value_range = fact.get("prediction_value_range")
        if isinstance(value_range, dict):
            summary = "prediction range: " + ", ".join(
                f"{key}={_render_fact_scalar(value_range[key])}" for key in ("min", "median", "max") if key in value_range
            )
        else:
            summary = "full prediction table written for delivery"
    return f"Provided table: {count} cases; target {target}; {prediction_type}; {summary}."


def _render_fact_scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def protocol_feedback_text(text: str) -> str:
    return f"Protocol feedback: {text}"


def protocol_feedback(messages: list[dict[str, Any]], text: str) -> None:
    messages.append({"role": "user", "content": protocol_feedback_text(text)})


# --- PS-OPSD privileged teacher (training only) ---------------------------------------------------
PRIVILEGED_SPECIFICATION_OPEN_TAG = "<privileged_training_specification>"
PRIVILEGED_SPECIFICATION_CLOSE_TAG = "</privileged_training_specification>"

PRIVILEGED_TEACHER_TRANSITION_TEXT = (
    "After understanding the privileged semantic specification, solve the original request "
    "using your own internal reasoning and the production tool protocol. Do not quote or reveal "
    "the privileged specification. Given the trajectory and tool feedback so far, choose the next "
    "valid tool action."
)


def privileged_teacher_block(annotation_json: str) -> str:
    """Render the PS-OPSD teacher's privileged block followed by the transition text.

    ``annotation_json`` must already be canonical
    (``json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))``) and contain no
    target value, benchmark answer, or gold TabPFN/judge output -- callers own that leakage boundary.
    """
    return (
        f"{PRIVILEGED_SPECIFICATION_OPEN_TAG}\n"
        f"{annotation_json}\n"
        f"{PRIVILEGED_SPECIFICATION_CLOSE_TAG}\n\n"
        f"{PRIVILEGED_TEACHER_TRANSITION_TEXT}"
    )
