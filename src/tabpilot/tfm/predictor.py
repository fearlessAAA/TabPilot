"""The frozen, plug-and-play tabular foundation model (TFM); no LLM or tool concerns.

``TFM_BACKENDS`` registers every supported TFM; ``tfm.backend`` in the config selects one and
:func:`predict_tfm` is the single entry point. TabPFN v3.5 (default) runs in this module; the
TabICLv2 and LimiX-2 adapters (``tabicl.py``, ``limix.py``) call only each library's documented
entry points with its official default settings.
"""

from __future__ import annotations

import ast
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from tabpilot.contracts import PredictionRecord, Scalar
from tabpilot.paths import DEFAULT_TFM_WEIGHTS_ROOT
from tabpilot.tfm.context import ModelContext, TabPFNFrameContractError, assert_tabpfn_preprocessing_contract

PredictionBackend = Literal[
    "tabpfn_v3_5_direct",
    "tabpfn_v3_5_many_class",
    "tabicl_v2_direct",
    "tabicl_v2_many_class",
    "limix_2_direct",
]
ProbabilitySemantics = Literal["native_tabpfn", "native_tfm", "many_class_decoded", "not_applicable"]


class TFMCapabilityError(ValueError):
    """The prepared request exceeds what the selected TFM officially supports (e.g. class count)."""


@dataclass(frozen=True)
class TFMBackend:
    """One selectable TFM: its checkpoint, context ceiling and officially supported feature kinds.

    ``max_context_rows``/``max_context_features`` are ceilings a config may tighten but never
    loosen. ``supports_datetime``/``supports_text`` state whether the TFM's official preprocessing
    uses date and free-text columns. ``max_classes`` is a hard class-count limit (``None``: any
    count, through the TFM's own many-class mechanism).
    """

    name: str
    model_version: str
    max_context_rows: int
    max_context_features: int
    supports_datetime: bool
    supports_text: bool
    max_classes: int | None = None


# TabPFN v3.5: checkpoint-native limits (``get_inference_config()``: 1M samples, 20k features,
# 160 native classes). TabICLv2 (tabicl 2.2.0): its README's validated range (up to 100k samples,
# 2k features); dates/text through the README's skrub TableVectorizer pipeline. LimiX-2 (limix
# 2.0.0): no text encoder and 2-10 classes only; dates through its official DatetimePreprocessing.
TFM_BACKENDS: dict[str, TFMBackend] = {
    "tabpfn_v3_5": TFMBackend(
        name="tabpfn_v3_5",
        model_version="v3.5",
        max_context_rows=1_000_000,
        max_context_features=20_000,
        supports_datetime=True,
        supports_text=True,
    ),
    "tabicl_v2": TFMBackend(
        name="tabicl_v2",
        model_version="tabicl-v2-20260212",
        max_context_rows=100_000,
        max_context_features=2_000,
        supports_datetime=True,
        supports_text=True,
    ),
    "limix_2": TFMBackend(
        name="limix_2",
        model_version="limix-2",
        max_context_rows=100_000,
        max_context_features=2_000,
        supports_datetime=True,
        supports_text=False,
        max_classes=10,
    ),
}
DEFAULT_TFM_BACKEND = "tabpfn_v3_5"


@dataclass(frozen=True)
class PredictionRun:
    records: list[PredictionRecord]
    model_version: str
    elapsed_seconds: float
    prediction_backend: PredictionBackend = "tabpfn_v3_5_direct"
    target_class_count: int | None = None
    native_class_limit: int | None = None
    probability_semantics: ProbabilitySemantics = "not_applicable"
    target_label_encoding: Literal["contiguous_integer_ids", "not_applicable"] = "not_applicable"
    many_class_config: dict[str, object] = field(default_factory=dict)
    # TabICL/LimiX: checkpoint file and the official preprocessing path that ran.
    backend_config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _TargetLabelCodec:
    """A history-only target codec for physically executable numeric classification."""

    labels: tuple[Scalar, ...]

    @classmethod
    def fit(cls, y: pd.Series) -> tuple["_TargetLabelCodec", np.ndarray]:
        codes, uniques = pd.factorize(y, sort=False)
        if bool((codes < 0).any()):
            raise ValueError("classification history target contains missing values")
        labels = tuple(_scalar(value) for value in uniques.tolist())
        if not labels:
            raise ValueError("classification history target has no labels")
        if len({str(label) for label in labels}) != len(labels):
            raise ValueError("target_label_string_collision")
        return cls(labels=labels), codes.astype(np.int64, copy=False)

    def decode(self, encoded: object) -> Scalar:
        numeric = float(encoded)
        index = int(numeric)
        if numeric != index or not 0 <= index < len(self.labels):
            raise ValueError("encoded class is outside the target label codec")
        return self.labels[index]


def predict_tfm(
    context: ModelContext,
    prediction_type: Literal["classification", "regression"],
    *,
    backend: str = DEFAULT_TFM_BACKEND,
    device: str = "cuda",
) -> PredictionRun:
    """Fit the registered TFM ``backend`` on the in-context history rows and predict the query rows."""
    if backend not in TFM_BACKENDS:
        raise ValueError(f"unknown TFM backend {backend!r}; choose one of {sorted(TFM_BACKENDS)}")
    profile = TFM_BACKENDS[backend]
    if context.datetime_columns and not profile.supports_datetime:
        raise ValueError(f"TFM backend {backend!r} cannot use date columns")
    if context.text_columns and not profile.supports_text:
        raise ValueError(f"TFM backend {backend!r} cannot use text columns")
    if prediction_type == "classification" and profile.max_classes is not None:
        class_count = int(context.y_history.nunique(dropna=True))
        if class_count > profile.max_classes:
            raise TFMCapabilityError(
                f"TFM backend {backend!r} supports at most {profile.max_classes} classes; the target has {class_count}"
            )
    if backend == "tabicl_v2":
        from tabpilot.tfm.tabicl import predict_tabicl

        return predict_tabicl(context, prediction_type, profile=profile, device=device)
    if backend == "limix_2":
        from tabpilot.tfm.limix import predict_limix

        return predict_limix(context, prediction_type, profile=profile, device=device)
    return _predict_tabpfn(context, prediction_type, profile=profile, device=device)


def tfm_weights_path(*parts: str) -> Path:
    """A TabICL/LimiX checkpoint under ``TABPILOT_TFM_WEIGHTS`` (default ``checkpoints/tfm``)."""
    path = Path(os.environ.get("TABPILOT_TFM_WEIGHTS", DEFAULT_TFM_WEIGHTS_ROOT)).joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"missing TFM checkpoint {path}; download it as described in the README")
    return path


def categorical_dtype_frames(context: ModelContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """History/query frames whose declared categorical columns carry pandas ``category`` dtype.

    TabICL and LimiX take no categorical-index argument; both encode ``category``/string columns
    themselves and treat numeric dtypes as numeric, so a categorical column with numeric codes
    would otherwise be continuous. The vocabulary is the sampled history's; context construction
    already guarantees every stated query category has a labeled history witness.
    """
    x_history = context.x_history.copy()
    x_query = context.x_query.copy()
    for index in context.categorical_features_indices:
        name = context.feature_columns[index]
        categories = pd.Index(pd.unique(x_history[name].dropna()))
        history = pd.Series(pd.Categorical(x_history[name], categories=categories), index=x_history.index)
        query = pd.Series(pd.Categorical(x_query[name], categories=categories), index=x_query.index)
        if bool((x_query[name].notna() & query.isna()).any()):
            raise TabPFNFrameContractError(
                f"categorical query value for {name!r} is outside the sampled history vocabulary"
            )
        x_history[name] = history
        x_query[name] = query
    return x_history, x_query


def _predict_tabpfn(
    context: ModelContext,
    prediction_type: Literal["classification", "regression"],
    *,
    profile: TFMBackend,
    device: str,
) -> PredictionRun:
    """TabPFN v3.5; the official many-class wrapper is used only above its native class alphabet."""
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    from tabpilot.tfm.compat import install_tabpfn_v3_5_qr_device_fix

    install_tabpfn_v3_5_qr_device_fix(device=device)
    version = ModelVersion(profile.model_version)
    started = perf_counter()
    # Prepare already ran the same CPU-only assertion. Rechecking directly
    # before model execution makes a stale/mutated context fail closed instead
    # of allowing TabPFN to silently coerce predict-time values to missing.
    assert_tabpfn_preprocessing_contract(context)
    if prediction_type == "classification":
        codec, encoded_y = _TargetLabelCodec.fit(context.y_history)
        model, backend, class_count, native_limit, probability_semantics, many_class_config = (
            _make_classifier(context, class_count=len(codec.labels), device=device, version=version)
        )
        fitted: dict[str, Any] = {}
        with _guard_predict_dtype_drift(context, fitted):
            model.fit(context.x_history, encoded_y)
            fitted["model"] = model
            encoded_labels = model.predict(context.x_query)
            probabilities = np.asarray(model.predict_proba(context.x_query), dtype=float)
        labels = [codec.decode(label) for label in encoded_labels]
        classes = [codec.decode(class_label) for class_label in model.classes_]
        _validate_classification_outputs(
            labels=labels,
            probabilities=probabilities,
            classes=classes,
            expected_rows=len(context.row_ids),
        )
        records = [
            PredictionRecord(
                row_id=row_id,
                predicted_label=_scalar(label),
                probabilities={
                    str(class_label): float(probability)
                    for class_label, probability in zip(classes, row_probabilities, strict=True)
                },
            )
            for row_id, label, row_probabilities in zip(
                context.row_ids,
                labels,
                probabilities,
                strict=True,
            )
        ]
        return PredictionRun(
            records=records,
            model_version=version.value.upper(),
            elapsed_seconds=perf_counter() - started,
            prediction_backend=backend,
            target_class_count=class_count,
            native_class_limit=native_limit,
            probability_semantics=probability_semantics,
            target_label_encoding="contiguous_integer_ids",
            many_class_config=many_class_config,
        )

    model = TabPFNRegressor.create_default_for_version(
        version,
        device=device,
        categorical_features_indices=context.categorical_features_indices,
        random_state=context.sampling_seed,
        **_native_modality_overrides(context),
    )
    fitted: dict[str, Any] = {}
    with _guard_predict_dtype_drift(context, fitted):
        model.fit(context.x_history, context.y_history)
        fitted["model"] = model
        values = model.predict(context.x_query)
    records = [
        PredictionRecord(row_id=row_id, predicted_value=float(value))
        for row_id, value in zip(context.row_ids, values, strict=True)
    ]
    return PredictionRun(
        records=records,
        model_version=version.value.upper(),
        elapsed_seconds=perf_counter() - started,
        prediction_backend="tabpfn_v3_5_direct",
    )


def _make_classifier(
    context: ModelContext,
    *,
    class_count: int,
    device: str,
    version: Any,
) -> tuple[Any, PredictionBackend, int, int, ProbabilitySemantics, dict[str, object]]:
    from tabpfn import TabPFNClassifier

    base = TabPFNClassifier.create_default_for_version(
        version,
        device=device,
        categorical_features_indices=context.categorical_features_indices,
        random_state=context.sampling_seed,
        **_native_modality_overrides(context),
    )
    native_limit = int(base.get_inference_config().MAX_NUMBER_OF_CLASSES)
    if class_count <= native_limit:
        return base, "tabpfn_v3_5_direct", class_count, native_limit, "native_tabpfn", {}

    from tabpfn_extensions.many_class import ManyClassClassifier

    many_class_config: dict[str, object] = {
        "alphabet_size": native_limit,
        "n_estimators_redundancy": 4,
        "random_state": context.sampling_seed,
    }
    return (
        ManyClassClassifier(estimator=base, **many_class_config),
        "tabpfn_v3_5_many_class",
        class_count,
        native_limit,
        "many_class_decoded",
        many_class_config,
    )


def _validate_classification_outputs(
    *,
    labels: Any,
    probabilities: np.ndarray,
    classes: list[object],
    expected_rows: int,
) -> None:
    if probabilities.ndim != 2 or probabilities.shape != (expected_rows, len(classes)):
        raise ValueError(
            "TFM classification probability shape does not match query rows and original classes."
        )
    if len(labels) != expected_rows:
        raise ValueError("TFM classification label count does not match query rows.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("TFM classification probabilities must be finite and non-negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("TFM classification probabilities must sum to one per row.")
    class_values = {_scalar(value) for value in classes}
    if any(_scalar(label) not in class_values for label in labels):
        raise ValueError("TFM classification label is outside the original history label space.")


def _scalar(value: object) -> str | int | float | bool:
    return value.item() if isinstance(value, np.generic) else value  # type: ignore[return-value]


_DTYPE_DRIFT_MARKER = "have a dtype at predict time that differs from fit time"


def _native_modality_overrides(context: ModelContext) -> dict[str, object]:
    """Dates arrive as tz-aware ``datetime64`` columns and text as pandas ``string`` columns
    (``context.py``); TabPFN expands them itself (calendar features; char n-gram tf-idf + SVD).
    Declared categorical columns are never treated as text by TabPFN."""
    return {"inference_config": {"TRANSFORM_DATES": True, "TRANSFORM_TEXT": True}}


@contextmanager
def _guard_predict_dtype_drift(context: ModelContext, fitted: dict[str, Any] | None = None):
    """Suppress only TabPFN's benign warning for wholly missing query columns.

    A sparse query can legitimately contain a categorical column that is native
    missing in every query row. TabPFN 8.1.0 may infer a different scalar dtype
    for that all-missing internal column and warn before coercing it back. No
    stated value can be lost in that case. Any warned column containing a
    non-missing query value is a real contract violation and remains fail-closed.
    """
    original_showwarning = warnings.showwarning

    def showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        text = str(message)
        if issubclass(category, UserWarning) and _DTYPE_DRIFT_MARKER in text:
            warned_columns = _dtype_warning_columns(text)
            expanded = _expanded_column_sources((fitted or {}).get("model"), context)
            if warned_columns and all(
                _query_column_missingness_is_contract_permitted(context, column, expanded)
                for column in warned_columns
            ):
                return
            raise TabPFNFrameContractError(
                "TabPFN fit/predict dtype contract drifted for a stated query value"
            )
        original_showwarning(message, category, filename, lineno, file=file, line=line)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = showwarning
        try:
            yield
        finally:
            warnings.showwarning = original_showwarning


def _dtype_warning_columns(message: str) -> list[int | str]:
    prefix = "Column(s) "
    suffix = " have a dtype at predict time"
    if prefix not in message or suffix not in message:
        return []
    payload = message.split(prefix, 1)[1].split(suffix, 1)[0]
    try:
        values = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, (int, str))]


def _tabpfn_encoder_column_order(context: ModelContext) -> list[str]:
    """The column name at each position TabPFN's fitted ``OrdinalEncoder`` actually sees.

    TabPFN expands ``datetime_columns``/``text_columns`` itself and never runs the ordinal-encoder
    path over them, so a dtype-drift warning's column position indexes this reduced column order,
    not the full ``context.x_query``.
    """
    native = set(context.datetime_columns) | set(context.text_columns)
    return [name for name in context.feature_columns if name not in native]


def _expanded_column_sources(model: Any, context: ModelContext) -> list[str]:
    """Source column of every feature TabPFN appended after the numeric/categorical block, in order.

    TabPFN expands each date column into calendar
    features (``DateTransformer``) and each text column into n-gram features (``TextTransformer``) and
    appends them after the untouched columns, dates first (tabpfn/preprocessing/datetimes.py,
    text.py ``_drop_and_append``). How many features a column yields is only known after fitting
    (skrub drops features a column cannot vary in), so it is read from the fitted transformers'
    ``fitted_columns_[position].output_names``. A dtype-drift warning's column position can point into
    this appended region; resolving it to its source column lets the guard apply the same
    all-missing-in-the-query rule as for an ordinary column. Returns ``[]`` when nothing was expanded or
    the model exposes no fitted transformers, which leaves such positions unresolved (fail closed).
    """
    date_transformer = getattr(model, "date_transformer_", None)
    text_transformer = getattr(model, "text_transformer_", None)
    date_fitted = dict(getattr(date_transformer, "fitted_columns_", None) or {})
    text_fitted = dict(getattr(text_transformer, "fitted_columns_", None) or {})
    if not date_fitted and not text_fitted:
        return []
    frame_columns = list(context.feature_columns)  # the frame handed to fit, before any expansion
    sources: list[str] = []
    for position in sorted(date_fitted):
        sources += [frame_columns[position]] * len(date_fitted[position].output_names)
    # TextTransformer runs on the DateTransformer output: date columns removed, order otherwise kept.
    after_dates = [name for index, name in enumerate(frame_columns) if index not in date_fitted]
    for position in sorted(text_fitted):
        sources += [after_dates[position]] * len(text_fitted[position].output_names)
    return sources


def _resolve_dtype_warning_column(
    context: ModelContext, column: int | str, expanded_sources: list[str] | None = None
) -> str | None:
    """Map a dtype-drift warning's column reference to its name in ``context.x_query``."""
    if isinstance(column, str):
        return column if column in context.x_query.columns else None
    order = _tabpfn_encoder_column_order(context)
    if 0 <= column < len(order):
        return order[column]
    offset = column - len(order)
    if expanded_sources and 0 <= offset < len(expanded_sources):
        return expanded_sources[offset]
    return None


def _query_column_is_fully_missing(
    context: ModelContext, column: int | str, expanded_sources: list[str] | None = None
) -> bool:
    feature = _resolve_dtype_warning_column(context, column, expanded_sources)
    if feature is None:
        return False
    return bool(context.x_query[feature].isna().all())


def _query_column_missingness_is_contract_permitted(
    context: ModelContext, column: int | str, expanded_sources: list[str] | None = None
) -> bool:
    """Permit only native missingness already checked by context construction.

    In addition to wholly sparse model-authored columns, a bound candidate table
    may contain exact categorical cells that were legally demoted to missing
    because no target-labeled fit witness exists.  Context asserts that no other
    value loss occurred before TabPFN runs, so this guard remains fail-closed for
    every unlisted cell and every non-categorical dtype drift.
    """
    if _query_column_is_fully_missing(context, column, expanded_sources):
        return True
    feature = _resolve_dtype_warning_column(context, column, expanded_sources)
    if feature is None:
        return False
    return any(cell.feature == feature for cell in context.provided_unrepresentable_category_cells)
