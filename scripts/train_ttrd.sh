#!/usr/bin/env bash
# TTRD: LoRA fine-tuning of the Response Agent on the teacher responses of one fold.
# Usage: scripts/train_ttrd.sh <fold> <output-dir>
#   Treatment-effect analysis records are repeated three times; MODEL defaults to Qwen/Qwen3.5-4B.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
fold=${1:?usage: $0 <fold> <output-dir>}
output=${2:?usage: $0 <fold> <output-dir>}
shift 2
mkdir -p "$output"
python - "$repo/data/ttrd/fold$fold.jsonl" "$output/train.jsonl" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as source, open(sys.argv[2], "w", encoding="utf-8") as target:
    for line in source:
        record = json.loads(line)
        copies = 3 if record["task"] == "treatment_effect_analysis" else 1
        target.writelines([json.dumps({"messages": record["messages"]}, ensure_ascii=False) + "\n"] * copies)
PY
export USE_HF=1 TOKENIZERS_PARALLELISM=false
python -m swift.cli.sft \
  --model "${MODEL:-Qwen/Qwen3.5-4B}" \
  --dataset "$output/train.jsonl" \
  --tuner_type lora --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --target_modules all-linear \
  --torch_dtype bfloat16 --attn_impl sdpa --gradient_checkpointing true \
  --enable_thinking false --add_non_thinking_prefix true --loss_scale default+ignore_empty_think \
  --max_length 8192 --truncation_strategy delete \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler_type cosine --warmup_ratio 0.0 --num_train_epochs 1 \
  --save_strategy epoch --save_only_model true --save_total_limit 1 \
  --logging_steps 1 --seed 0 --data_seed 0 --dataset_num_proc 1 --dataloader_num_workers 0 \
  --output_dir "$output" "$@"
