"""TabPilot: Execution Agent -> frozen TFM -> Response Agent, for one natural-language request."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from tabpilot.agents.execution_agent import ExecutionAgent, ExecutionResult
from tabpilot.agents.response_agent import ResponseAgent, prediction_facts
from tabpilot.config import TabPilotConfig
from tabpilot.contracts import RunTrace
from tabpilot.data.safe_schema import build_safe_schema, project_safe_schema_for_model
from tabpilot.data.topbench import load_task_bundle
from tabpilot.llm.client import ToolCallingClient
from tabpilot.tools.inspect_history import InspectHistoryTool
from tabpilot.tools.materialization import materialize_current_table_artifacts
from tabpilot.tools.prepare_tfm_input import PrepareTFMInputTool
from tabpilot.tools.registry import ToolRegistry
from tabpilot.tools.run_store import RunStore
from tabpilot.tools.submit_provided_table_intent import SubmitProvidedTableIntentTool
from tabpilot.tools.tfm_predict import TFMPredictTool

RANKING_ANSWER = "Predictions were generated for {rows} rows of the provided table; see the prediction table."


class TabPilot:
    """Runs one request end to end and writes ``<output_dir>/<case>/<run_id>/trace.json``.

    A request that comes with a candidate table (``current.csv``) takes the Ranking and Filtering
    path, whose output is the materialized prediction table; every other request takes the
    Predictive Reasoning path and ends with the Response Agent's answer.
    """

    def __init__(
        self,
        config: TabPilotConfig,
        *,
        execution_client: ToolCallingClient,
        response_client: ToolCallingClient,
        device: str = "cuda",
    ) -> None:
        self.config = config
        self.device = device
        self.execution_agent = ExecutionAgent(
            client=execution_client, sampling=config.execution_sampling, max_turns=config.max_turns, seed=config.seed
        )
        self.response_agent = ResponseAgent(
            client=response_client, sampling=config.response_sampling, seed=config.seed, turn_offset=config.max_turns
        )

    def run(
        self,
        *,
        task_dir: Path,
        query_file: Path,
        output_dir: Path,
        case_key: str | None = None,
        provenance: dict[str, object] | None = None,
    ) -> RunTrace:
        bundle = load_task_bundle(task_dir, query_file)
        run_id = f"run_{uuid4().hex}"
        schema = build_safe_schema(
            bundle.history, task_dir, dropped_columns=bundle.dropped_columns, current=bundle.current
        )
        schema_view = project_safe_schema_for_model(schema)
        store = RunStore(
            run_id=run_id,
            query=bundle.query,
            history=bundle.history,
            schema=schema,
            current=bundle.current,
            device=self.device,
            max_context_rows=self.config.max_context_rows,
            max_context_features=self.config.max_context_features,
            random_state=self.config.seed,
            tfm_backend=self.config.tfm_backend,
            model_features=self.config.model_features,
        )
        ranking = bundle.current is not None
        if ranking:
            registry = ToolRegistry([SubmitProvidedTableIntentTool()], store)
            execution = self.execution_agent.run_ranking(bundle.query_id, bundle.query, registry, schema_view)
        else:
            registry = ToolRegistry([InspectHistoryTool(), PrepareTFMInputTool(), TFMPredictTool()], store)
            execution = self.execution_agent.run_predictive_reasoning(
                bundle.query_id, bundle.query, registry, schema_view
            )
        trace = self._finish(run_id, bundle.query_id, bundle.query, execution, store, ranking)
        if ranking and trace.status == "success":
            prediction = store.latest_prediction()
            materialization = materialize_current_table_artifacts(
                output_dir=output_dir / Path(case_key or bundle.query_id) / run_id,
                history_columns=list(bundle.history.columns),
                target_column=prediction.target_column,
                current=bundle.current,
                prediction=prediction,
            )
            trace = trace.model_copy(
                update={
                    "provenance": {**trace.provenance, "execution": {"materialization": materialization}},
                    "materialized_table_path": materialization["prediction_table_path"],
                }
            )
        trace = trace.model_copy(
            update={
                "case_key": case_key,
                "provenance": {
                    **trace.provenance,
                    **(provenance or {}),
                    "execution_path": "ranking_and_filtering" if ranking else "predictive_reasoning",
                },
            }
        )
        _write_trace(trace, output_dir)
        return trace

    def _finish(
        self,
        run_id: str,
        query_id: str,
        query: str,
        execution: ExecutionResult,
        store: RunStore,
        ranking: bool,
    ) -> RunTrace:
        def trace(status: str, **fields: object) -> RunTrace:
            return RunTrace(
                run_id=run_id,
                query_id=query_id,
                status=status,  # type: ignore[arg-type]
                events=execution.events,
                protocol_events=execution.protocol_events,
                role_events=execution.role_events,
                prediction_ids=store.prediction_ids,
                **fields,
            )

        if execution.failure_code is not None:
            return trace("failure", failure_code=execution.failure_code)
        prediction = store.latest_prediction()
        prepared = store.get_prepared(prediction.prepared_id) if prediction is not None else None
        if prediction is None or prepared is None:
            return trace("failure", failure_code="runtime_state_error")
        if ranking:
            # The Ranking and Filtering output is the prediction table; no Response Agent call.
            return trace("success", answer=RANKING_ANSWER.format(rows=len(prediction.records)))
        facts = prediction_facts(prepared.rows, prediction.model_dump(mode="json"))
        answer, protocol_events, role_events = self.response_agent.respond(query_id, query, facts)
        execution.protocol_events.extend(protocol_events)
        execution.role_events.extend(role_events)
        if answer is None:
            return trace("failure", failure_code="empty_final_answer")
        return trace("success", answer=answer)


def _write_trace(trace: RunTrace, output_dir: Path) -> None:
    case_path = Path(trace.case_key or trace.query_id)
    if case_path.is_absolute() or ".." in case_path.parts:
        raise ValueError("case_key must be a relative path without parent traversal")
    run_dir = output_dir / case_path / trace.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trace.json").write_text(
        json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
