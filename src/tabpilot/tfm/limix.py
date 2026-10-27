"""LimiX-2 through its official ``LimiXPredictor`` (``limix`` 2.0.0) and default v2 configs.

The packaged ``cls_default_noretrieval_v2.json`` / ``reg_default_noretrieval_v2.json`` are used
as shipped. LimiX label-encodes ``category``/string columns itself and models NaN natively.
Dates: when the context carries date columns, every pipeline member additionally enables the
official ``DatetimePreprocessing`` switch of ``inference/v2_0/predictor.py`` (train-only
detection and fit, the only mode it accepts); the shipped configs leave it off, and datetime64
input raises without it. LimiX has no text encoder and accepts 2-10 classes only; the backend
profile declares this, so ``predict_tfm`` rejects such requests before this adapter runs.
RMSNorm: the checkpoint's ``config.rmsnorm_impl`` is switched from ``triton`` to LimiX's own
``torch`` implementation (same math, passed through the official ``ckpt=`` argument): the Triton
QK-norm kernel launches one program per row and at ~100k context rows exceeds the GPU launch
limit on some devices, leaving outputs unwritten.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal

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

LIMIX_CHECKPOINT = "LimiX-2.ckpt"
LIMIX_CLASSIFICATION_CONFIG = "cls_default_noretrieval_v2.json"
LIMIX_REGRESSION_CONFIG = "reg_default_noretrieval_v2.json"
LIMIX_RMSNORM_IMPL = "torch"
LIMIX_DATETIME_PREPROCESSING = {"enabled": True, "detection_scope": "train", "fit_scope": "train"}


def predict_limix(
    context: ModelContext,
    prediction_type: Literal["classification", "regression"],
    *,
    profile: TFMBackend,
    device: str,
) -> PredictionRun:
    import torch

    from tabpilot.tfm.compat import install_limix_cache_dir_fix

    install_limix_cache_dir_fix()
    # Official switch: reuse the loaded weights across calls in one process.
    os.environ.setdefault("LIMIX_REUSE_FROZEN_MODEL", "1")
    from limix import LimiXPredictor

    started = perf_counter()
    assert_tabpfn_preprocessing_contract(context)
    if context.text_columns:
        raise ValueError("LimiX-2 has no text preprocessing")
    x_history, x_query = categorical_dtype_frames(context)
    config_name = LIMIX_CLASSIFICATION_CONFIG if prediction_type == "classification" else LIMIX_REGRESSION_CONFIG
    datetime_preprocessing = bool(context.datetime_columns)
    model_path = str(tfm_weights_path("limix", LIMIX_CHECKPOINT))
    # A fresh predictor per call, so results never depend on call order.
    predictor = LimiXPredictor(
        device=torch.device(device),
        model_path=model_path,
        ckpt=limix_checkpoint(model_path),
        inference_config=limix_inference_config(config_name, datetime_preprocessing=datetime_preprocessing),
        seed=context.sampling_seed,
    )
    backend_config: dict[str, object] = {
        "checkpoint": LIMIX_CHECKPOINT,
        "inference_config": config_name,
        "datetime_preprocessing": datetime_preprocessing,
        "rmsnorm_impl": LIMIX_RMSNORM_IMPL,
    }
    if prediction_type == "classification":
        codec, encoded_y = _TargetLabelCodec.fit(context.y_history)
        probabilities = np.asarray(
            predictor.predict(x_history, encoded_y, x_query, task_type="Classification"), dtype=float
        )
        # LimiX label-encodes y itself; contiguous codes 0..k-1 keep its column order = code order.
        class_codes = sorted(set(encoded_y.tolist()))
        classes = [codec.decode(code) for code in class_codes]
        if probabilities.ndim != 2 or probabilities.shape[1] != len(class_codes):
            raise ValueError("LimiX classification probability shape does not match the history classes.")
        labels = [classes[int(index)] for index in probabilities.argmax(axis=1)]
        _validate_classification_outputs(
            labels=labels, probabilities=probabilities, classes=classes, expected_rows=len(context.row_ids)
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
            for row_id, label, row_probabilities in zip(context.row_ids, labels, probabilities, strict=True)
        ]
        return PredictionRun(
            records=records,
            model_version=profile.model_version,
            elapsed_seconds=perf_counter() - started,
            prediction_backend="limix_2_direct",
            target_class_count=len(classes),
            native_class_limit=profile.max_classes,
            probability_semantics="native_tfm",
            target_label_encoding="contiguous_integer_ids",
            backend_config=backend_config,
        )

    values = np.asarray(
        predictor.predict(x_history, context.y_history.to_numpy(dtype=float), x_query, task_type="Regression"),
        dtype=float,
    ).reshape(-1)
    if values.shape != (len(context.row_ids),) or not np.isfinite(values).all():
        raise ValueError("LimiX regression output must be one finite value per query row.")
    return PredictionRun(
        records=[
            PredictionRecord(row_id=row_id, predicted_value=float(value))
            for row_id, value in zip(context.row_ids, values, strict=True)
        ],
        model_version=profile.model_version,
        elapsed_seconds=perf_counter() - started,
        prediction_backend="limix_2_direct",
        backend_config=backend_config,
    )


@lru_cache(maxsize=1)
def limix_checkpoint(model_path: str) -> dict:
    """The LimiX-2 checkpoint with its official ``rmsnorm_impl`` set to the torch implementation."""
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint["config"]["rmsnorm_impl"] = LIMIX_RMSNORM_IMPL
    return checkpoint


def limix_inference_config(name: str, *, datetime_preprocessing: bool) -> str | dict:
    """The shipped config path, or the same config with official date preprocessing on every member."""
    path = _packaged_config(name)
    if not datetime_preprocessing:
        return str(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    for member in config["pipelines"]:
        member["DatetimePreprocessing"] = dict(LIMIX_DATETIME_PREPROCESSING)
    return config


def _packaged_config(name: str) -> Path:
    import config as limix_config_package  # LimiX installs its JSON configs as top-level ``config``

    return Path(limix_config_package.__file__).parent / name
