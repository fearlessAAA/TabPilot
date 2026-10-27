"""Typed semantic submission for a runtime-bound candidate table."""

from __future__ import annotations

from tabpilot.agents.prompts import SUBMIT_PROVIDED_TABLE_INTENT_TOOL_DESCRIPTION
from tabpilot.contracts import (
    PrepareTFMInputArguments,
    ProvidedTableIntentArguments,
    ToolError,
    ToolResponse,
)
from tabpilot.tools.prepare_tfm_input import PrepareTFMInputTool
from tabpilot.tools.run_store import RunStore


class SubmitProvidedTableIntentTool:
    """Prepare the bound table after the model submits only target and type.

    This is deliberately a one-shot semantic boundary. It does not expose
    candidate rows, and it never runs the model; the runtime invokes prediction
    only after this deterministic preparation succeeds.
    """

    name = "submit_provided_table_intent"
    description = SUBMIT_PROVIDED_TABLE_INTENT_TOOL_DESCRIPTION
    input_model = ProvidedTableIntentArguments

    def invoke(self, arguments: ProvidedTableIntentArguments, store: RunStore) -> ToolResponse:
        if store.current is None:
            return ToolResponse(
                status="invalid",
                errors=[
                    ToolError(
                        code="provided_table_required",
                        path="target_column",
                        message="This semantic submission requires a runtime-bound candidate table.",
                        retryable=False,
                    )
                ],
            )
        prepared = PrepareTFMInputTool().invoke(
            PrepareTFMInputArguments(
                target_column=arguments.target_column,
                prediction_type=arguments.prediction_type,
                rows=[],
            ),
            store,
        )
        if prepared.status != "ok":
            return prepared
        # The prepare tool already returns a bounded, value-free observation
        # for provided tables. Keep only fields needed by the semantic caller;
        # current rows never enter the model-visible response.
        data = dict(prepared.data or {})
        data.pop("normalized_rows", None)
        return prepared.model_copy(update={"data": data})
