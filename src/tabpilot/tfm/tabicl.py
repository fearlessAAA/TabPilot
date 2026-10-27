"""TabICLv2 through its official scikit-learn estimators (``tabicl`` 2.2.0), default settings.

Date and free-text columns are only usable through the README's recommended pipeline
``make_pipeline(TableVectorizer(low_cardinality="passthrough"), TabICL...)`` (skrub): TabICL's
built-in preprocessing ordinal-encodes string/category columns, mean-imputes numeric NaN, and
silently drops ``datetime64`` columns. The pipeline is used exactly when the prepared context
carries date or text columns; otherwise the estimator receives the frame directly.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal

import numpy as np

from tabpilot.contracts import PredictionRecord
from tabpilot.tfm.context import ModelContext, assert_tabpfn_preprocessing_contract
from tabpilot.tfm.predictor import (
    PredictionRun,
    TFMBackend,
    _scalar,
    _TargetLabelCodec,
    _validate_classification_outputs,
    categorical_dtype_frames,
    tfm_weights_path,
)

TABICL_CLASSIFIER_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
TABICL_REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"


def predict_tabicl(
    context: ModelContext,
    prediction_type: Literal["classification", "regression"],
    *,
    profile: TFMBackend,
    device: str,
) -> PredictionRun:
    from tabicl import TabICLClassifier, TabICLRegressor

    started = perf_counter()
    assert_tabpfn_preprocessing_contract(context)
    x_history, x_query = categorical_dtype_frames(context)
    use_vectorizer = bool(context.datetime_columns or context.text_columns)
    checkpoint = TABICL_CLASSIFIER_CHECKPOINT if prediction_type == "classification" else TABICL_REGRESSOR_CHECKPOINT
    common: dict[str, Any] = {
        "model_path": str(tfm_weights_path("tabicl", checkpoint)),
        "checkpoint_version": checkpoint,
        "allow_auto_download": False,
        "device": device,
        "random_state": context.sampling_seed,
    }
    config: dict[str, object] = {
        "checkpoint": checkpoint,
        "preprocessing": "skrub_table_vectorizer" if use_vectorizer else "tabicl_builtin",
    }
    if prediction_type == "classification":
        codec, encoded_y = _TargetLabelCodec.fit(context.y_history)
        estimator = TabICLClassifier(**common)
        model = _with_vectorizer(estimator) if use_vectorizer else estimator
        model.fit(x_history, encoded_y)
        encoded_labels = model.predict(x_query)
        probabilities = np.asarray(model.predict_proba(x_query), dtype=float)
        labels = [codec.decode(label) for label in encoded_labels]
        classes = [codec.decode(class_label) for class_label in estimator.classes_]
        _validate_classification_outputs(
            labels=labels, probabilities=probabilities, classes=classes, expected_rows=len(context.row_ids)
        )
        native_limit = int(estimator.model_.max_classes)
        class_count = len(codec.labels)
        records = [
            PredictionRecord(
                row_id=row_id,
                predicted_label=_scalar(label),
                probabilities={
                    str(class_label): float(probability)
                    for class_label, probability in zip(classes, row_probabilities, strict=True)
                },
            )
            for row_id, label, row_probabilities in zip(context.row_ids, labels, probabilities, strict=True)
        ]
        return PredictionRun(
            records=records,
            model_version=profile.model_version,
            elapsed_seconds=perf_counter() - started,
            prediction_backend="tabicl_v2_many_class" if class_count > native_limit else "tabicl_v2_direct",
            target_class_count=class_count,
            native_class_limit=native_limit,
            probability_semantics="native_tfm",
            target_label_encoding="contiguous_integer_ids",
            backend_config=config,
        )

    estimator = TabICLRegressor(**common)
    model = _with_vectorizer(estimator) if use_vectorizer else estimator
    model.fit(x_history, context.y_history.to_numpy(dtype=float))
    values = np.asarray(model.predict(x_query), dtype=float)
    if values.shape != (len(context.row_ids),) or not np.isfinite(values).all():
        raise ValueError("TabICL regression output must be one finite value per query row.")
    return PredictionRun(
        records=[
            PredictionRecord(row_id=row_id, predicted_value=float(value))
            for row_id, value in zip(context.row_ids, values, strict=True)
        ],
        model_version=profile.model_version,
        elapsed_seconds=perf_counter() - started,
        prediction_backend="tabicl_v2_direct",
        backend_config=config,
    )


def _with_vectorizer(estimator: Any) -> Any:
    from sklearn.pipeline import make_pipeline
    from skrub import TableVectorizer

    return make_pipeline(TableVectorizer(low_cardinality="passthrough"), estimator)
