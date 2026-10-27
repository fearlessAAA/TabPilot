#!/usr/bin/env bash
# PS-OPSD: LoRA on-policy distillation of the Execution Agent from the privileged teacher.
# Usage: scripts/train_ps_opsd.sh <data-dir> <output-dir>
#   <data-dir> is the output of scripts/build_training_data.py; MODEL (default Qwen/Qwen3.5-4B)
#   is the backbone of both the student and the teacher.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
data=${1:?usage: $0 <data-dir> <output-dir>}
output=${2:?usage: $0 <data-dir> <output-dir>}
shift 2
model=${MODEL:-Qwen/Qwen3.5-4B}

export TABPILOT_TASKS="$data/tasks.jsonl"
export TABPILOT_MAX_TOKENS_PER_TURN=4096 TABPILOT_MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768} TABPILOT_MAX_TURNS=10
export TABPILOT_PRESENCE_PENALTY=1.5 TABPILOT_KD_TEMPERATURE=1.1 TABPILOT_KL_CLIP=0.05
export TABPILOT_ROLLOUT_LOG="$output/rollouts.jsonl"
export USE_HF=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export RANK=${RANK:-0} LOCAL_RANK=${LOCAL_RANK:-0} WORLD_SIZE=${WORLD_SIZE:-1} LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29500}

python -m tabpilot.training.launch \
  --rlhf_type gkd \
  --model "$model" --teacher_model "$model" \
  --tuner_type lora --lora_rank 64 --lora_alpha 128 --lora_dropout 0.0 \
  --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --dataset "$data/ps_opsd.jsonl" --remove_unused_columns false \
  --external_plugins "$repo/src/tabpilot/training/ms_swift_plugin.py" \
  --beta 0 --lmbda 1.0 --sft_alpha 0 --num_generations 1 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 --max_steps "${MAX_STEPS:-100}" \
  --optim adamw_torch --weight_decay 0 --warmup_ratio 0.0 --lr_scheduler_type linear --max_grad_norm 0.1 \
  --torch_dtype bfloat16 --attn_impl sdpa --padding_free false --gradient_checkpointing true \
  --agent_template qwen3_5 --enable_thinking false --template_backend jinja \
  --temperature 0.7 --top_p 0.8 --top_k 20 --repetition_penalty 1.0 \
  --max_length "$TABPILOT_MAX_MODEL_LEN" --max_completion_length "$TABPILOT_MAX_MODEL_LEN" \
  --use_vllm true --vllm_mode colocate --vllm_max_model_len "$TABPILOT_MAX_MODEL_LEN" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.10}" --vllm_max_num_seqs 8 \
  --vllm_enable_prefix_caching false --vllm_enforce_eager true \
  --sleep_level 1 --offload_model true --offload_optimizer true --offload_teacher_model false \
  --seed 42 --data_seed 42 --report_to tensorboard --logging_steps 1 \
  --save_strategy steps --save_steps "${MAX_STEPS:-100}" --save_total_limit 1 --save_only_model true \
  --output_dir "$output" "$@"
