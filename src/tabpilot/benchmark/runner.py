"""Resumable runs of TabPilot over the TopBench requests, tracked in ``manifest.jsonl``."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from tabpilot.paths import REPOSITORY_ROOT

PREDICTIVE_REASONING_TASKS = ("single_point_prediction", "decision_making", "treatment_effect_analysis")
RANKING_TASK = "ranking_and_filtering"
TASKS = (*PREDICTIVE_REASONING_TASKS, RANKING_TASK)
# TopBench names each task directory with a code suffix (``<Table>_B1``, ...); its scorers use the codes.
TOPBENCH_TASK_CODES = {
    "single_point_prediction": "B1",
    "decision_making": "B2",
    "treatment_effect_analysis": "B3",
    "ranking_and_filtering": "B4",
}
_NON_QUERY_FILENAMES = {"info.json", "info_mod.json", "current.json"}


@dataclass(frozen=True)
class ManifestCase:
    case_key: str
    task: str
    query_path: str
    task_dir: str
    status: str = "pending"
    failure_code: str | None = None
    failure_detail: str | None = None
    run_id: str | None = None
    trace_path: str | None = None

    @property
    def task_code(self) -> str:
        return TOPBENCH_TASK_CODES[self.task]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManifestCase":
        return cls(**{name: payload.get(name) for name in cls.__dataclass_fields__ if name in payload})


def enumerate_cases(benchmark_root: Path, tasks: Iterable[str] = TASKS) -> list[ManifestCase]:
    """List every request file of the selected tasks without reading any answer field."""
    root = Path(benchmark_root).resolve()
    cases: list[ManifestCase] = []
    for task in tasks:
        task_root = root / task
        if not task_root.exists():
            raise FileNotFoundError(task_root)
        suffix = f"_{TOPBENCH_TASK_CODES[task]}"
        for query_path in sorted(task_root.rglob("*.json")):
            if query_path.name in _NON_QUERY_FILENAMES:
                continue
            task_dir = query_path.parent
            if task == RANKING_TASK:
                # Ranking requests live in ``<Table>_B4/<Table>_B4_00k/<...>_current.json``.
                if not query_path.stem.endswith("_current") or suffix not in task_dir.parent.name:
                    continue
            elif not query_path.stem[-1:].isdigit() or not task_dir.name.endswith(suffix):
                continue
            cases.append(
                ManifestCase(
                    case_key=str(query_path.relative_to(root).with_suffix("")),
                    task=task,
                    query_path=str(query_path),
                    task_dir=str(task_dir),
                )
            )
    if not cases:
        raise RuntimeError(f"No TopBench request files found under {root}")
    return cases


def source_table(task_dir: str | Path) -> str:
    """The source table of a request; TopBench names task directories ``<Table>_B<k>[_<n>]``."""
    return re.sub(r"_B[1-4](?:_\d+)?$", "", Path(task_dir).name)


def fold_sources(split_path: Path, fold: int, partition: str) -> set[str]:
    """The ``train`` or ``test`` source tables of one cross-validation fold."""
    folds = json.loads(Path(split_path).read_text(encoding="utf-8"))["folds"]
    entry = next((item for item in folds if item["fold_id"] == fold), None)
    if entry is None:
        raise ValueError(f"fold {fold} is not in {split_path}")
    return set(entry[f"{partition}_sources"])


def write_manifest(path: Path, cases: Iterable[ManifestCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        "".join(json.dumps(asdict(case), ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_manifest(path: Path) -> list[ManifestCase]:
    return [
        ManifestCase.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_manifest(
    *,
    pipeline: Any,
    manifest_path: Path,
    output_root: Path,
    run_config: dict[str, object],
    resume: bool = True,
    on_case_done: Callable[[ManifestCase], None] | None = None,
) -> list[ManifestCase]:
    """Run every pending case; a crash is recorded as a failure and the run continues."""
    cases = read_manifest(manifest_path)
    done: list[ManifestCase] = []
    for case in cases:
        if resume and case.status in {"success", "failure"}:
            done.append(case)
            continue
        try:
            trace = pipeline.run(
                task_dir=Path(case.task_dir),
                query_file=Path(case.query_path),
                output_dir=output_root / "traces",
                case_key=case.case_key,
                provenance=run_config,
            )
            update = {
                "status": trace.status,
                "failure_code": trace.failure_code,
                "run_id": trace.run_id,
                "trace_path": str(output_root / "traces" / case.case_key / trace.run_id / "trace.json"),
            }
        except Exception as error:  # noqa: BLE001 - every case stays in the manifest
            update = {
                "status": "failure",
                "failure_code": f"runtime_exception:{type(error).__name__}",
                "failure_detail": repr(error)[:2_000],
            }
        done.append(ManifestCase(**{**asdict(case), **update}))
        write_manifest(manifest_path, [*done, *cases[len(done) :]])
        if on_case_done is not None:
            on_case_done(done[-1])
    return done


def run_provenance(settings: dict[str, object]) -> dict[str, object]:
    """What a run depends on: settings, prompt and tool-schema hashes, code revision, packages."""
    from tabpilot.agents import prompts
    from tabpilot.contracts import (
        InspectHistoryArguments,
        PrepareTFMInputArguments,
        ProvidedTableIntentArguments,
        TFMPredictArguments,
    )

    prompt_text = "\n".join(
        value for name, value in sorted(vars(prompts).items()) if name.isupper() and isinstance(value, str)
    )
    tool_schemas = {
        "inspect_history": InspectHistoryArguments.model_json_schema(),
        "prepare_tfm_input": PrepareTFMInputArguments.model_json_schema(),
        "submit_provided_table_intent": ProvidedTableIntentArguments.model_json_schema(),
        "tfm_predict": TFMPredictArguments.model_json_schema(),
    }
    return {
        **settings,
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "tool_schema_sha256": hashlib.sha256(json.dumps(tool_schemas, sort_keys=True).encode()).hexdigest(),
        "code_revision": _git_revision(),
        "package_versions": {name: _package_version(name) for name in ("tabpfn", "tabpfn-extensions", "tabicl", "limix", "vllm")},
    }


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
