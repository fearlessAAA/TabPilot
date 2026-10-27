"""Score a finished TabPilot run with the official TopBench evaluators.

* Predictive Reasoning answers are written in TopBench's inference layout and scored by its judge
  (``eval_b2_v1.py``, DeepSeek API, ``DEEPSEEK_API_KEY``), which extracts structured predictions
  from the free-text answers.
* Ranking and Filtering prediction tables are scored by TopBench's deterministic CSV scorer
  (``eval_b4_v3.py``).

Every request stays in the denominator: a failed or unscored request scores 0, except NMAE, which
scores 1.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from tabpilot.benchmark.runner import PREDICTIVE_REASONING_TASKS, RANKING_TASK, ManifestCase, read_manifest
from tabpilot.paths import TOPBENCH_ROOT

MODEL_NAME = "tabpilot"
MODE = "with_tool"
# (metric name, TopBench breakdown key) per Predictive Reasoning task.
PREDICTIVE_METRICS = {
    "single_point_prediction": [("acc", "accuracy")],
    "decision_making": [("acc", "avg_prediction"), ("dec", "decision")],
    "treatment_effect_analysis": [("acc", "pred_002_accuracy"), ("trend", "trend_accuracy")],
}


def evaluate(run_root: Path, benchmark_root: Path, *, workers: int = 8) -> dict[str, Any]:
    # The TopBench scripts run with their output directory as cwd, so relative paths would break.
    run_root, benchmark_root = Path(run_root).resolve(), Path(benchmark_root).resolve()
    cases = read_manifest(run_root / "manifest.jsonl")
    output = run_root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    predictive = [case for case in cases if case.task in PREDICTIVE_REASONING_TASKS]
    if predictive:
        metrics.update(_evaluate_predictive_reasoning(predictive, benchmark_root, output / "predictive_reasoning", workers))
    ranking = [case for case in cases if case.task == RANKING_TASK]
    if ranking:
        metrics[RANKING_TASK] = _evaluate_ranking(ranking, benchmark_root, output / "ranking_and_filtering", workers)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _evaluate_predictive_reasoning(
    cases: list[ManifestCase], benchmark_root: Path, root: Path, workers: int
) -> dict[str, Any]:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY must be set (see .env.example) to run the TopBench judge")
    payloads = {}
    for case in cases:
        path = _layout_path(root, case, benchmark_root, f"{Path(case.query_path).stem}_{MODEL_NAME}_{MODE}.json")
        payload = json.loads(Path(case.query_path).read_text(encoding="utf-8"))
        payload.update(response=_answer(case), csv_success="N/A", performance_metrics={})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payloads[case.case_key] = path
    data_view_module = _topbench_module("topbench.compatibility.legacy_data_view")
    script = TOPBENCH_ROOT / "src/topbench/compatibility/legacy_scripts/eval_b2_v1.py"
    for task in sorted({case.task for case in cases}):
        with tempfile.TemporaryDirectory(prefix="tabpilot_topbench_") as data_view:
            data_view_module.prepare_legacy_data_view(benchmark_root.resolve(), data_view, tasks=[task])
            _run(
                [sys.executable, str(script), "--inference_root", str(root), "--benchmark",
                 _code(task), "--models", MODEL_NAME, "--mode", MODE, "--workers", str(workers)],
                env={"TOPBENCH_LEGACY_DATA_ROOT": str(Path(data_view).resolve())},
                cwd=root,
                log=root / f"judge_{task}.log",
            )
    metrics: dict[str, Any] = {}
    for task, names in PREDICTIVE_METRICS.items():
        task_cases = [case for case in cases if case.task == task]
        if not task_cases:
            continue
        totals = dict.fromkeys([name for name, _ in names], 0.0)
        for case in task_cases:
            evaluation = payloads[case.case_key].with_name(payloads[case.case_key].stem + "_eval.json")
            breakdown = _read_json(evaluation).get("breakdown", {}) if case.status == "success" else {}
            for name, key in names:
                totals[name] += float((breakdown.get(key) or {}).get("score") or 0.0)
        metrics[task] = {name: total / len(task_cases) for name, total in totals.items()} | {"n": len(task_cases)}
    return metrics


def _evaluate_ranking(cases: list[ManifestCase], benchmark_root: Path, root: Path, workers: int) -> dict[str, Any]:
    for case in cases:
        table = _prediction_table(case)
        if table is None:
            continue
        target = _layout_path(root, case, benchmark_root, "prediction_current.csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(table, target)
        shutil.copyfile(case.query_path, target.parent / Path(case.query_path).name)
    data_view_module = _topbench_module("topbench.compatibility.legacy_data_view")
    script = TOPBENCH_ROOT / "src/topbench/compatibility/legacy_scripts/eval_b4_v3.py"
    with tempfile.TemporaryDirectory(prefix="tabpilot_topbench_") as data_view:
        prepared = data_view_module.prepare_legacy_data_view(benchmark_root.resolve(), Path(data_view), tasks=[RANKING_TASK])
        _run(
            [sys.executable, str(script), "--inference_root", str(root.resolve()), "--dataset_root",
             str(prepared[RANKING_TASK].resolve()), "--models", MODEL_NAME, "--mode", MODE, "--workers", str(workers)],
            env={},
            cwd=root,
            log=root / "scorer.log",
        )
    details = _read_json(root / MODEL_NAME / MODE / "B4_details.json") or []
    by_task = {record.get("task"): record for record in details if isinstance(record, dict)}
    totals = {"regression": {"recall": [], "ndcg": [], "nmae": []}, "classification": {"f1": []}}
    for case in cases:
        record = by_task.get(Path(case.task_dir).name) if case.status == "success" else None
        task_type = _ranking_task_type(case)
        for metric, values in totals[task_type].items():
            fallback = 1.0 if metric == "nmae" else 0.0
            values.append(float(record[metric]) if record and record.get(metric) is not None else fallback)
    return {
        "regression": {metric: sum(values) / len(values) for metric, values in totals["regression"].items() if values}
        | {"n": len(totals["regression"]["recall"])},
        "classification": {"f1": _mean(totals["classification"]["f1"]), "n": len(totals["classification"]["f1"])},
    }


def _layout_path(root: Path, case: ManifestCase, benchmark_root: Path, filename: str) -> Path:
    relative_dir = Path(case.task_dir).resolve().relative_to((benchmark_root / case.task).resolve())
    return root / MODEL_NAME / MODE / _code(case.task) / relative_dir / filename


def _code(task: str) -> str:
    return ManifestCase(case_key="", task=task, query_path="", task_dir="").task_code


def _ranking_task_type(case: ManifestCase) -> str:
    for name in ("info_mod.json", "info.json"):
        info = _read_json(Path(case.task_dir).parent / name) or _read_json(Path(case.task_dir) / name)
        if isinstance(info, dict) and info.get("task_type") in {"regression", "classification"}:
            return info["task_type"]
    raise RuntimeError(f"cannot determine the task type of {case.case_key}")


def _answer(case: ManifestCase) -> str:
    trace = _read_json(Path(case.trace_path)) if case.status == "success" and case.trace_path else None
    answer = (trace or {}).get("answer")
    return answer if isinstance(answer, str) else ""


def _prediction_table(case: ManifestCase) -> Path | None:
    trace = _read_json(Path(case.trace_path)) if case.status == "success" and case.trace_path else None
    path = (trace or {}).get("materialized_table_path")
    return Path(path) if isinstance(path, str) and Path(path).exists() else None


def _topbench_module(name: str) -> Any:
    source = str(TOPBENCH_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return __import__(name, fromlist=["_"])


def _run(command: list[str], *, env: dict[str, str], cwd: Path, log: Path) -> None:
    environment = {**os.environ, **env, "PYTHONPATH": f"{TOPBENCH_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}"}
    cwd.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"TopBench evaluator failed ({result.returncode}); see {log}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
