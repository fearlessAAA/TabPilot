# TabPilot

**TabPilot: Empowering Small Language Models for Natural-Language Tabular Prediction via
Declarative Query Compilation, Privileged Distillation, and Structure-Aware Rewards**

Natural-language tabular prediction asks for outcomes that are not stored in a table and must be
inferred from its historical rows. TabPilot is an agentic framework that couples a 4B small
language model (SLM) with a **plug-and-play frozen tabular foundation model (TFM)** through
**declarative predictive query compilation**:

* the **Execution Agent** iteratively compiles each request into a structured prediction query
  `Q = (target, type, scenarios)` and validates it through leakage-safe tools;
* the frozen **TFM** executes the query by in-context inference, without downstream training;
* the **Response Agent** synthesizes the TFM predictions into a grounded answer.

Both agents are trained for unseen tables with a three-stage local-to-global curriculum:

1. **Privileged Semantic On-Policy Self-Distillation (PS-OPSD)**: dense token-level supervision on
   the agent's own trajectories from a frozen teacher conditioned on privileged query semantics;
2. **Structure-Aware GRPO (SA-GRPO)**: reinforcement learning with an outcome-label-free,
   TFM-agnostic structural reward;
3. **Teacher-Guided Tabular Response Distillation (TTRD)**: faithful comparative synthesis for the
   Response Agent.

On TopBench, the 4B TabPilot outperforms frontier proprietary and 671B open-weight baselines in
prediction accuracy and transfers zero-shot across unseen TFM backbones, while using far fewer
tokens than existing paradigms. This repository contains inference, evaluation on TopBench and
the three training stages; the trained agents are on Hugging Face at
[XiaoboX/TabPilot](https://huggingface.co/XiaoboX/TabPilot).

## Repository layout

```
configs/tabpilot.yaml            inference settings (sampling, turn budget, TFM backend and context rows)
src/tabpilot/
  cli.py                         `tabpilot predict | run | evaluate`
  pipeline.py                    one request end to end: Execution Agent -> TFM -> Response Agent
  agents/                        Execution Agent, Response Agent, prompts, tool-loop protocol
  tools/                         InspectHistory, PrepareTFMInput, TFMPredict, SubmitProvidedTableIntent
  tfm/                           TFM context construction; TabPFN v3.5 (default), TabICLv2, LimiX-2
  data/                          TopBench request loading, history cleaning, safe schema
  llm/client.py                  OpenAI-compatible tool-calling client (vLLM)
  benchmark/                     resumable TopBench runs and scoring with the TopBench evaluators
  training/
    reference_query.py           reference query Q* from the TopBench annotations
    structure_aware_reward.py    R = V * I[target] * I[type] * S (cell F1 + Hungarian matching)
    rollout.py                   multi-turn tool environment for on-policy training (ms-swift)
    ps_opsd.py                   PS-OPSD: per-turn privileged-teacher distillation, clipped forward KL
    sa_grpo.py                   SA-GRPO: reward, group resampling, no-signal step skipping
    ms_swift_plugin.py           registers the above with ms-swift (`--external_plugins`)
    launch.py                    `swift rlhf` entry point
scripts/
  build_training_data.py         PS-OPSD / SA-GRPO data of one fold
  train_ps_opsd.sh               PS-OPSD (LoRA)
  merge_lora.py                  merge the PS-OPSD adapter before SA-GRPO
  train_sa_grpo.sh               SA-GRPO (full parameters)
  train_ttrd.sh                  TTRD Response Agent LoRA
data/
  reference_query_rows.json      scenarios to predict per request (defines Q*'s rows)
  splits.json                    five-fold cross-validation over TopBench source tables
  ttrd/fold*.jsonl               TTRD data (teacher: Gemini 3.8 Flash)
  NOTICE                         TopBench attribution and license for these files
checkpoints/                     downloaded models (see checkpoints/README.md)
benchmarks/topbench/             TopBench code and data (not included; see below)
```

## Installation

TabPilot needs Python 3.12 and a GPU. The pinned versions in `pyproject.toml` are the ones used
for the paper.

```bash
git clone https://github.com/fearlessAAA/TabPilot.git && cd TabPilot
conda create -n tabpilot python=3.12 -y && conda activate tabpilot
pip install -e ".[serve]"           # inference and evaluation
pip install -e ".[serve,train]"     # additionally, training
pip install -e ".[serve,tfm]"       # optional: TabICLv2 and LimiX-2 as alternative TFMs
cp .env.example .env                # then fill in TABPFN_TOKEN and DEEPSEEK_API_KEY
```

`pip install vllm` installs the CUDA build of vLLM and PyTorch. On AMD GPUs, install vLLM's ROCm
build first (see the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html))
and then TabPilot with `pip install -e ".[train]"`.

`.env` holds `TABPFN_TOKEN` (needed once to download the TabPFN v3.5 weights) and
`DEEPSEEK_API_KEY` (TopBench's answer judge, used by `tabpilot evaluate`).

## TopBench

Download the publicly available [TopBench](https://github.com/LAMDA-Tabular/TopBench). Put its code and data under `benchmarks/topbench/`:

TabPilot reads requests from `benchmarks/topbench/data/` (override with `--benchmark-root`) and calls the TopBench evaluators in `benchmarks/topbench/src/`. It never writes into that directory.

## Inference

### 1. Serve the agents

Both agents are served by vLLM with an OpenAI-compatible API. Download the trained agents of a
fold (see `checkpoints/README.md`; each fold has its own models):

```bash
hf download XiaoboX/TabPilot --include "execution-agent/fold0/*" --include "response-agent/fold0/*" \
  --local-dir checkpoints
```

Start the Execution Agent, wait until it is ready, then start the Response Agent (two vLLM
servers that start at the same time on one GPU miscount its free memory). The memory fractions
below suit one 80 GB GPU shared with the TFM; adjust them to your GPU.

```bash
# Execution Agent on port 8000
vllm serve checkpoints/execution-agent/fold0 --served-model-name tabpilot-execution --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.55 --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 &
until curl -fs http://127.0.0.1:8000/health; do sleep 5; done

# Response Agent on port 8001: the base model plus the TTRD adapter
vllm serve Qwen/Qwen3.5-4B --served-model-name tabpilot-response-base --port 8001 \
  --max-model-len 32768 --gpu-memory-utilization 0.20 --language-model-only \
  --enable-lora --max-lora-rank 64 --lora-modules tabpilot-response=checkpoints/response-agent/fold0 &
until curl -fs http://127.0.0.1:8001/health; do sleep 5; done
```

Untrained agents: serve `Qwen/Qwen3.5-4B` as both `tabpilot-execution` and `tabpilot-response`.
Other servers or names: `--execution-url/--execution-model` and `--response-url/--response-model`.

### 2. Run

```bash
# One request
tabpilot predict --query-file benchmarks/topbench/data/single_point_prediction/<...>/<request>.json

# TopBench: the test tables of fold 0, all four tasks (resumable; rerun the same command to continue)
tabpilot run --output-root outputs/fold0 --fold 0

# Selected tasks or requests
tabpilot run --output-root outputs/single_point --tasks single_point_prediction --case-list my_cases.txt
```

`tabpilot run` writes `manifest.jsonl` (per-request status), `run_config.json`, and one
`traces/<request>/<run_id>/trace.json` per request, holding every tool call, observation and
answer; Ranking and Filtering requests also write `prediction_table.csv`. Settings are in
`configs/tabpilot.yaml` (`--config` selects another file).

### Switching the TFM

The TFM is plug-and-play: the trained agents stay the same and only `tfm.backend` in
`configs/tabpilot.yaml` changes. **The default, used for all main results, is TabPFN v3.5
(`tabpfn_v3_5`) with up to 100,000 history rows.**

| `tfm.backend` | Model | Max. history rows | Dates / text | Classes |
|---|---|---|---|---|
| `tabpfn_v3_5` (default) | TabPFN v3.5 | 1,000,000 | yes / yes | any |
| `tabicl_v2` | TabICLv2 | 100,000 | yes / yes | any |
| `limix_2` | LimiX-2 | 100,000 | yes / no | 2-10 |

TabICLv2 and LimiX-2 need `pip install -e ".[tfm]"` and their weights under `checkpoints/tfm/`
(or set `TABPILOT_TFM_WEIGHTS` to another directory):

```bash
hf download jingang/TabICL tabicl-classifier-v2-20260212.ckpt tabicl-regressor-v2-20260212.ckpt \
  --local-dir checkpoints/tfm/tabicl
hf download stable-ai/LimiX-2 LimiX-2.ckpt --local-dir checkpoints/tfm/limix   # non-commercial license
```

Then copy the config, set its backend and pass it to `tabpilot`:

```bash
sed 's/backend: tabpfn_v3_5/backend: limix_2/' configs/tabpilot.yaml > configs/tabpilot_limix_2.yaml
tabpilot run --config configs/tabpilot_limix_2.yaml --output-root outputs/fold0_limix_2 --fold 0
```

Each TFM runs through its official preprocessing. Columns a TFM cannot use (for LimiX-2: free
text and string identifiers) are left out of its input and reported to the Execution Agent as a
warning; `tfm.model_features` can narrow the column kinds further. A target with more classes than
the TFM supports fails with `tfm_capability_exceeded`. `run_config.json` and every trace record
the TFM that ran.

### 3. Evaluate

```bash
tabpilot evaluate --run-root outputs/fold0
```

Predictive Reasoning answers are scored by TopBench's judge (DeepSeek-V3.2 through the DeepSeek
API; for another OpenAI-compatible provider set `DEEPSEEK_BASE_URL` and `DEEPSEEK_JUDGE_MODEL_ID`
in `.env`), Ranking and Filtering tables by TopBench's table scorer. Scores per task are printed and written to
`outputs/fold0/evaluation/metrics.json`; every request counts, a failed request scores 0 (NMAE 1).

## Training

Each stage is trained per fold on that fold's training tables. The scripts take the backbone from
`MODEL` (default `Qwen/Qwen3.5-4B`) and pass extra arguments through to ms-swift; GPU memory
settings `VLLM_GPU_MEMORY_UTILIZATION` and `MAX_MODEL_LEN` (default 32768) can be set in the environment.

```bash
# Execution Agent data for fold 0 (CPU): ps_opsd.jsonl, sa_grpo.jsonl, tasks.jsonl
python scripts/build_training_data.py --fold 0 --output-dir outputs/train/fold0/data

# 1. PS-OPSD (LoRA, 100 steps), then merge the adapter
scripts/train_ps_opsd.sh outputs/train/fold0/data outputs/train/fold0/ps_opsd
python scripts/merge_lora.py --base Qwen/Qwen3.5-4B \
  --adapter outputs/train/fold0/ps_opsd/<version>/checkpoint-100 --output outputs/train/fold0/ps_opsd_merged

# 2. SA-GRPO (full parameters, one epoch)
scripts/train_sa_grpo.sh outputs/train/fold0/data outputs/train/fold0/ps_opsd_merged outputs/train/fold0/sa_grpo

# 3. TTRD Response Agent (LoRA, independent of the Execution Agent)
scripts/train_ttrd.sh 0 outputs/train/fold0/ttrd
```

The final SA-GRPO checkpoint is served as the Execution Agent and the TTRD adapter as the
Response Agent's LoRA module. During training the Execution Agent runs the real
`inspect_history` and `prepare_tfm_input` tools; the TFM is not called. Both Execution stages
append one JSON line per rollout to `<output>/rollouts.jsonl` (termination, reward terms for
SA-GRPO).

## Data in this repository

* `data/reference_query_rows.json` — for every TopBench Predictive Reasoning request, which
  annotated scenarios are sent to the TFM: the single scenario (single-point prediction), every
  candidate (decision making), or the changed scenario plus the reference scenario when the
  request does not state its outcome (treatment-effect analysis). With TopBench's annotated
  target, type and features it defines the reference query used by PS-OPSD and SA-GRPO.
* `data/splits.json` — the five folds; all requests from one source table share a fold.
* `data/ttrd/fold<k>.jsonl` — TTRD records of fold `k`'s training tables: the Response Agent's
  input (request and TFM predictions of the reference query) and the teacher's answer.

## License

MIT (see `LICENSE`). The files in `data/` are derived from TopBench (MIT); see `data/NOTICE`.

## Acknowledgments
This work benefits from the previous excellent work of [TopBench](https://github.com/LAMDA-Tabular/TopBench), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [TabPFN3.5](https://arxiv.org/abs/2609.17895), and [ms-swift
](https://github.com/modelscope/ms-swift).
