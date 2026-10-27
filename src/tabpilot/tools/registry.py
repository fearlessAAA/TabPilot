"""OpenAI-function schema generation and safe local dispatch."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from tabpilot.agents.prompts import misnested_features_message
from tabpilot.contracts import ToolError, ToolResponse
from tabpilot.tools.base import AgentTool
from tabpilot.tools.run_store import RunStore


class ToolRegistry:
    def __init__(self, tools: Iterable[AgentTool], store: RunStore) -> None:
        tool_list = list(tools)
        self._tools = {tool.name: tool for tool in tool_list}
        if len(self._tools) != len(tool_list):
            raise ValueError("tool names must be unique")
        self._store = store

    @property
    def store(self) -> RunStore:
        return self._store

    def openai_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def invoke(self, name: str, raw_arguments: str) -> ToolResponse:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResponse(
                status="invalid",
                errors=[ToolError(code="unknown_tool", message=f"Tool {name!r} is not registered.")],
            )
        try:
            arguments = tool.input_model.model_validate_json(raw_arguments)
        except ValidationError as error:
            return ToolResponse(
                status="invalid",
                errors=_validation_errors(
                    tool_name=name,
                    raw_arguments=raw_arguments,
                    error=error,
                    store=self._store,
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            return ToolResponse(
                status="invalid",
                errors=[ToolError(code="invalid_tool_arguments", message=str(error))],
            )
        try:
            return tool.invoke(arguments, self._store)
        except Exception as error:  # CUDA/model/filesystem failures are terminal system failures.
            return ToolResponse(
                status="error",
                errors=[ToolError(code="tool_system_error", message=f"{type(error).__name__}: {error}", retryable=False)],
            )


def _validation_errors(
    *,
    tool_name: str,
    raw_arguments: str,
    error: ValidationError,
    store: RunStore,
) -> list[ToolError]:
    """Turn flattened prediction rows into one actionable structural repair per row."""
    grouped, grouped_locations = _misnested_feature_errors(
        tool_name=tool_name,
        raw_arguments=raw_arguments,
        schema_names=set(store.schema.names),
    )
    remaining = [
        ToolError(
            code="invalid_tool_arguments",
            path=".".join(str(part) for part in item["loc"]),
            message=str(item["msg"]),
        )
        for item in error.errors()
        if tuple(item["loc"]) not in grouped_locations
    ]
    return [*grouped, *remaining]


def _misnested_feature_errors(
    *, tool_name: str, raw_arguments: str, schema_names: set[str]
) -> tuple[list[ToolError], set[tuple[object, ...]]]:
    if tool_name not in {"prepare_tfm_input", "submit_prediction_program", "submit_prediction_context"}:
        return [], set()
    try:
        payload = json.loads(raw_arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], set()
    if not isinstance(payload, dict):
        return [], set()
    if tool_name == "prepare_tfm_input":
        rows = payload.get("rows")
        prefix: tuple[object, ...] = ("rows",)
        if not isinstance(rows, list):
            return [], set()
        errors: list[ToolError] = []
        locations: set[tuple[object, ...]] = set()
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            columns = sorted(key for key in row if key not in {"row_id", "features"} and key in schema_names)
            if not columns:
                continue
            locations.update((*prefix, row_index, column) for column in columns)
            errors.append(
                ToolError(
                    code="misnested_features",
                    path=".".join(str(part) for part in (*prefix, row_index, "features")),
                    message=misnested_features_message(row_index, columns),
                )
            )
        return errors, locations
    elif tool_name == "submit_prediction_program":
        program = payload.get("program")
        rows = program.get("prediction_set", {}).get("rows") if isinstance(program, dict) else None
        prefix = ("program", "prediction_set", "rows")
    else:
        context = payload.get("context")
        rows = context.get("rows") if isinstance(context, dict) else None
        prefix = ("context", "rows")
    if not isinstance(rows, list):
        return [], set()

    errors: list[ToolError] = []
    locations: set[tuple[object, ...]] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        columns = sorted(
            key for key in row if key not in {"row_id", "features"} and key in schema_names
        )
        if not columns:
            continue
        locations.update((*prefix, row_index, column) for column in columns)
        errors.append(
            ToolError(
                code="misnested_features",
                path=".".join(str(part) for part in (*prefix, row_index, "features")),
                message=misnested_features_message(row_index, columns),
            )
        )
    return errors, locations
