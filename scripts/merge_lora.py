"""Merge a LoRA adapter into its base model (PS-OPSD -> SA-GRPO initialization).

Usage: python scripts/merge_lora.py --base Qwen/Qwen3.5-4B --adapter <checkpoint> --output <dir>
"""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Resolve model ids on Hugging Face, like the training scripts (ms-swift defaults to ModelScope).
    os.environ.setdefault("USE_HF", "1")
    from swift.model import get_model_processor

    model, processor = get_model_processor(args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).merge_and_unload(safe_merge=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(args.output)


if __name__ == "__main__":
    main()
