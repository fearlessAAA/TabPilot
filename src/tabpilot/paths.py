"""Repository-relative default locations."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# The TopBench repository (code and data) is placed here; TabPilot never writes into it.
TOPBENCH_ROOT = REPOSITORY_ROOT / "benchmarks" / "topbench"
DEFAULT_BENCHMARK_ROOT = TOPBENCH_ROOT / "data"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "tabpilot.yaml"
DATA_ROOT = REPOSITORY_ROOT / "data"
DEFAULT_SPLITS = DATA_ROOT / "splits.json"
# TabICL and LimiX checkpoints (``tfm.backend: tabicl_v2 | limix_2``); override with TABPILOT_TFM_WEIGHTS.
DEFAULT_TFM_WEIGHTS_ROOT = REPOSITORY_ROOT / "checkpoints" / "tfm"
