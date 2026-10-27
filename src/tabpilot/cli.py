"""Command-line entry point: ``tabpilot predict | run | evaluate``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from tabpilot.benchmark.runner import (
    TASKS,
    enumerate_cases,
    fold_sources,
    read_manifest,
    run_manifest,
    run_provenance,
    source_table,
    write_manifest,
)
from tabpilot.config import load_config
from tabpilot.llm.client import OpenAICompatibleToolClient
from tabpilot.paths import DEFAULT_BENCHMARK_ROOT, DEFAULT_CONFIG, DEFAULT_SPLITS, REPOSITORY_ROOT
from tabpilot.pipeline import TabPilot


def main() -> None:
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(prog="tabpilot", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict", help="Answer one TopBench request.")
    predict.add_argument("--query-file", type=Path, required=True, help="A TopBench request JSON file.")
    predict.add_argument("--output-dir", type=Path, default=Path("outputs/predict"))
    _add_agent_arguments(predict)

    run = commands.add_parser("run", help="Run TabPilot over TopBench (resumable).")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    run.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    run.add_argument("--case-list", type=Path, help="Run only the case keys listed in this file (one per line).")
    run.add_argument("--fold", type=int, help="Run only the test source tables of this cross-validation fold.")
    run.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    _add_agent_arguments(run)

    evaluate = commands.add_parser("evaluate", help="Score a finished run with the TopBench evaluators.")
    evaluate.add_argument("--run-root", type=Path, required=True)
    evaluate.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    evaluate.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    if args.command == "evaluate":
        from tabpilot.benchmark.evaluate import evaluate as evaluate_run

        print(json.dumps(evaluate_run(args.run_root, args.benchmark_root, workers=args.workers), indent=2))
        return

    config = load_config(args.config)
    pipeline = TabPilot(
        config,
        execution_client=OpenAICompatibleToolClient(base_url=args.execution_url, model=args.execution_model),
        response_client=OpenAICompatibleToolClient(base_url=args.response_url, model=args.response_model),
        device=args.device,
    )
    if args.command == "predict":
        trace = pipeline.run(task_dir=args.query_file.parent, query_file=args.query_file, output_dir=args.output_dir)
        print(trace.answer if trace.status == "success" else f"failed: {trace.failure_code}")
        return

    manifest_path = args.output_root / "manifest.jsonl"
    if not manifest_path.exists():
        cases = enumerate_cases(args.benchmark_root, args.tasks)
        if args.case_list:
            wanted = set(args.case_list.read_text(encoding="utf-8").split())
            cases = [case for case in cases if case.case_key in wanted]
        if args.fold is not None:
            sources = fold_sources(args.splits, args.fold, "test")
            cases = [case for case in cases if source_table(case.task_dir) in sources]
        write_manifest(manifest_path, cases)
    settings = {
        "config": config.as_dict(),
        "execution_model": args.execution_model,
        "response_model": args.response_model,
    }
    (args.output_root / "run_config.json").write_text(json.dumps(run_provenance(settings), indent=2), encoding="utf-8")
    done = run_manifest(
        pipeline=pipeline,
        manifest_path=manifest_path,
        output_root=args.output_root,
        run_config=settings,
        on_case_done=lambda case: print(f"{case.case_key}: {case.status} {case.failure_code or ''}", flush=True),
    )
    succeeded = sum(case.status == "success" for case in done)
    print(f"{succeeded}/{len(read_manifest(manifest_path))} requests succeeded")


def _add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execution-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--execution-model", default="tabpilot-execution")
    parser.add_argument("--response-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--response-model", default="tabpilot-response")
    parser.add_argument("--device", default="cuda", help="Device of the TFM.")


if __name__ == "__main__":
    main()
