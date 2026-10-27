"""Build the Execution Agent training data of one cross-validation fold (CPU only).

Writes to ``--output-dir``:

* ``ps_opsd.jsonl`` — PS-OPSD prompts: the agent's initial messages and tool schema, plus the
  teacher's privileged specification (``teacher_prompt``);
* ``sa_grpo.jsonl`` — SA-GRPO prompts (requests whose reference query is a valid query);
* ``tasks.jsonl`` — per request: task directory, request file and the canonical reference query,
  read by the training environment (``TABPILOT_TASKS``);
* ``summary.json`` — counts, and the requests left out of SA-GRPO with the reason.

Only the fold's training source tables are used, and only Predictive Reasoning requests.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tabpilot.agents.prompts import PRIVILEGED_SPECIFICATION_OPEN_TAG, execution_messages
from tabpilot.benchmark.runner import PREDICTIVE_REASONING_TASKS, enumerate_cases, fold_sources, source_table
from tabpilot.contracts import PrepareTFMInputArguments
from tabpilot.data.safe_schema import build_safe_schema, project_safe_schema_for_model
from tabpilot.data.topbench import load_task_bundle
from tabpilot.paths import DATA_ROOT, DEFAULT_BENCHMARK_ROOT, DEFAULT_SPLITS
from tabpilot.tools.inspect_history import InspectHistoryTool
from tabpilot.tools.prepare_tfm_input import PrepareTFMInputTool
from tabpilot.tools.registry import ToolRegistry
from tabpilot.tools.run_store import RunStore
from tabpilot.tools.tfm_predict import TFMPredictTool
from tabpilot.training.reference_query import (
    ReferenceQueryUnavailable,
    build_reference_query,
    canonical_json,
    load_reference_rows,
    privileged_specification,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--reference-rows", type=Path, default=DATA_ROOT / "reference_query_rows.json")
    args = parser.parse_args()

    sources = fold_sources(args.splits, args.fold, "train")
    reference_rows = load_reference_rows(args.reference_rows)
    ps_opsd, sa_grpo, tasks, excluded = [], [], [], []
    for case in enumerate_cases(args.benchmark_root, PREDICTIVE_REASONING_TASKS):
        if source_table(case.task_dir) not in sources:
            continue
        task_dir, query_file = Path(case.task_dir), Path(case.query_path)
        bundle = load_task_bundle(task_dir, query_file)
        schema = build_safe_schema(bundle.history, task_dir, dropped_columns=bundle.dropped_columns, current=bundle.current)
        store = RunStore(run_id=case.case_key, query=bundle.query, history=bundle.history, schema=schema, device="cpu")
        registry = ToolRegistry([InspectHistoryTool(), PrepareTFMInputTool(), TFMPredictTool()], store)
        prompt = {
            "task_id": case.case_key,
            "messages": execution_messages(bundle.query, project_safe_schema_for_model(schema)),
            "tools": registry.openai_definitions(),
        }
        predict = list(reference_rows[case.case_key]["predict"])
        specification = privileged_specification(query_file, predict)
        if PRIVILEGED_SPECIFICATION_OPEN_TAG in json.dumps(prompt):
            raise AssertionError(f"privileged content in the student prompt of {case.case_key}")
        ps_opsd.append({**prompt, "teacher_prompt": canonical_json(specification)})
        record = {"task_id": case.case_key, "task": case.task, "task_dir": case.task_dir, "query_path": case.query_path}
        try:
            reference = build_reference_query(
                query_file=query_file,
                task=case.task,
                predict=predict,
                schema=schema,
                history=bundle.history,
                dropped_columns=list(bundle.dropped_columns),
            )
            # The reference query must itself pass prepare_tfm_input, context construction included.
            prepared = PrepareTFMInputTool().invoke(
                PrepareTFMInputArguments(
                    target_column=reference.target_column,
                    prediction_type=reference.prediction_type,
                    rows=reference.prediction_rows(),
                ),
                store,
            )
            if prepared.status != "ok":
                raise ReferenceQueryUnavailable("reference_query_rejected", str([error.code for error in prepared.errors]))
        except ReferenceQueryUnavailable as error:
            excluded.append({"task_id": case.case_key, "reason": error.reason, "detail": str(error)})
            tasks.append({**record, "reference_query": None})
            continue
        sa_grpo.append(prompt)
        tasks.append({**record, "reference_query": reference.to_json()})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("ps_opsd.jsonl", ps_opsd), ("sa_grpo.jsonl", sa_grpo), ("tasks.jsonl", tasks)):
        with open(args.output_dir / name, "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    summary = {
        "fold": args.fold,
        "ps_opsd_requests": len(ps_opsd),
        "sa_grpo_requests": len(sa_grpo),
        "requests_by_task": dict(Counter(task["task"] for task in tasks)),
        "excluded_from_sa_grpo": excluded,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "excluded_from_sa_grpo"} | {"excluded": len(excluded)}))


if __name__ == "__main__":
    main()
