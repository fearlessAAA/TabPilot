"""Prediction-only tool: prepared context in, TFM predictions out."""

from __future__ import annotations

from uuid import uuid4

from tabpilot.agents.prompts import TFM_CAPABILITY_FEEDBACK, TFM_DTYPE_CONTRACT_FEEDBACK, TFM_PREDICT_TOOL_DESCRIPTION
from tabpilot.tfm.context import TabPFNFrameContractError
from tabpilot.contracts import (
    PredictionArtifact,
    TFMPredictArguments,
    ToolError,
    ToolResponse,
)
from tabpilot.tfm.predictor import TFMCapabilityError, predict_tfm
from tabpilot.tools.run_store import RunStore


class TFMPredictTool:
    name = "tfm_predict"
    description = TFM_PREDICT_TOOL_DESCRIPTION
    input_model = TFMPredictArguments

    def invoke(self, arguments: TFMPredictArguments, store: RunStore) -> ToolResponse:
        prepared = store.get_prepared(arguments.prepared_id)
        if prepared is None:
            return ToolResponse(
                status="invalid",
                errors=[
                    ToolError(
                        code="unknown_prepared_id",
                        path="prepared_id",
                        message="prepared_id is not available in this run; call prepare_tfm_input first.",
                    )
                ],
            )
        try:
            prediction_run = predict_tfm(
                prepared.context, prepared.prediction_type, backend=store.tfm_backend, device=store.device
            )
        except TFMCapabilityError:
            return ToolResponse(
                status="error",
                errors=[ToolError(code="tfm_capability_exceeded", message=TFM_CAPABILITY_FEEDBACK, retryable=False)],
            )
        except TabPFNFrameContractError:
            return ToolResponse(
                status="error",
                errors=[
                    ToolError(
                        code="tfm_dtype_contract",
                        message=TFM_DTYPE_CONTRACT_FEEDBACK,
                        retryable=False,
                    )
                ],
            )
        artifact = PredictionArtifact(
            prediction_id=f"pred_{uuid4().hex}",
            prepared_id=prepared.prepared_id,
            target_column=prepared.target_column,
            prediction_type=prepared.prediction_type,
            records=prediction_run.records,
            model_version=prediction_run.model_version,
            elapsed_seconds=prediction_run.elapsed_seconds,
            history_rows_used=prepared.context.history_rows_used,
            feature_count=len(prepared.context.feature_columns),
            sampling_strategy=prepared.context.sampling_strategy,
            sampling_seed=prepared.context.sampling_seed,
            prediction_backend=prediction_run.prediction_backend,
            target_class_count=prediction_run.target_class_count,
            native_class_limit=prediction_run.native_class_limit,
            probability_semantics=prediction_run.probability_semantics,
            target_label_encoding=prediction_run.target_label_encoding,
            many_class_config=prediction_run.many_class_config,
            backend_config=prediction_run.backend_config,
        )
        store.put_prediction(artifact)
        return ToolResponse(
            status="ok",
            data=artifact.model_dump(mode="json"),
            provenance={
                "sampled_row_hash": prepared.context.sampled_row_hash,
                "prediction_backend": prediction_run.prediction_backend,
                "target_class_count": prediction_run.target_class_count,
                "native_class_limit": prediction_run.native_class_limit,
                "probability_semantics": prediction_run.probability_semantics,
                "target_label_encoding": prediction_run.target_label_encoding,
                "many_class_config": prediction_run.many_class_config,
                "backend_config": prediction_run.backend_config,
                "datetime_columns": list(prepared.context.datetime_columns),
                "text_columns": list(prepared.context.text_columns),
            },
        )
