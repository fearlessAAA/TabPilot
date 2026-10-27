#!/usr/bin/env bash
# SA-GRPO: full-parameter GRPO of the Execution Agent with the Structure-Aware Reward.
# Usage: scripts/train_sa_grpo.sh <data-dir> <init-model> <output-dir>
#   <init-model> is the PS-OPSD model with its adapter merged (scripts/merge_lora.py).
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
data=${1:?usage: $0 <data-dir> <init-model> <output-dir>}
model=${2:?usage: $0 <data-dir> <init-model> <output-dir>}
output=${3:?usage: $0 <data-dir> <init-model> <output-dir>}
shift 3

export TABPILOT_TASKS="$data/tasks.jsonl"
export TABPILOT_MAX_TOKENS_PER_TURN=4096 TABPILOT_MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768} TABPILOT_PRESENCE_PENALTY=1.5
export TABPILOT_ROLLOUT_LOG="$output/rollouts.jsonl"
export USE_HF=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0

python -m tabpilot.training.launch \
  --rlhf_type grpo \
  --model "$model" --model_type qwen3_5 --tuner_type full \
  --dataset "$data/sa_grpo.jsonl" --dataset_shuffle true --train_dataloader_shuffle true \
  --external_plugins "$repo/src/tabpilot/training/ms_swift_plugin.py" \
  --multi_turn_scheduler tabpilot_sa_grpo --max_turns 10 --completion_length_limit_scope per_round \
  --reward_funcs tabpilot_structure_aware_reward --reward_weights 1.0 \
  --callbacks tabpilot_skip_no_signal_steps \
  --loss_type grpo --epsilon 0.2 --epsilon_high 0.2 --beta 0 --scale_rewards none \
  --dynamic_sample true --max_resample_times 3 --overlong_filter false \
  --num_iterations 2 --num_generations 8 --generation_batch_size 8 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 --num_train_epochs 1 \
  --optim adamw_torch --weight_decay 0 --warmup_steps 0 --lr_scheduler_type cosine --max_grad_norm 1.0 \
  --torch_dtype bfloat16 --attn_impl sdpa --padding_free false --gradient_checkpointing true \
  --template qwen3_5_tabpilot --template_backend swift --agent_template qwen3_5 --enable_thinking false \
  --temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --repetition_penalty 1.0 \
  --max_length "$TABPILOT_MAX_MODEL_LEN" --max_completion_length "$TABPILOT_MAX_MODEL_LEN" \
  --use_vllm true --vllm_mode colocate --vllm_max_model_len "$TABPILOT_MAX_MODEL_LEN" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.10}" --vllm_max_num_seqs 8 \
  --vllm_enable_prefix_caching false --vllm_enforce_eager true \
  --sleep_level 1 --offload_model true --offload_optimizer true \
  --seed 42 --data_seed 42 --report_to tensorboard --logging_steps 1 \
  --save_strategy epoch --save_total_limit 1 --save_only_model true \
  --output_dir "$output" "$@"
