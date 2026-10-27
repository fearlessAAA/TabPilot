"""Interfaces for server-bound OpenAI function tools."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from tabpilot.contracts import ToolResponse


class AgentTool(Protocol):
    name: str
    description: str
    input_model: type[BaseModel]

    def invoke(self, arguments: BaseModel, store: "RunStore") -> ToolResponse: ...


from tabpilot.tools.run_store import RunStore  # noqa: E402

