# Checkpoints

The trained agents are released on Hugging Face at
[XiaoboX/TabPilot](https://huggingface.co/XiaoboX/TabPilot). There is one pair per
cross-validation fold `k` (`data/splits.json`); fold `k` was trained only on that fold's training
source tables, so evaluate it on fold `k`'s test tables (`tabpilot run --fold k`).

| Path | Contents |
|---|---|
| `execution-agent/fold<k>/` | Execution Agent: full Qwen3.5-4B weights after PS-OPSD → SA-GRPO |
| `response-agent/fold<k>/` | Response Agent: TTRD LoRA adapter for the `Qwen/Qwen3.5-4B` base model |

Download them into this directory:

```bash
hf download XiaoboX/TabPilot --include "execution-agent/fold0/*" --include "response-agent/fold0/*" \
  --local-dir checkpoints
```

Files in this directory other than this README are ignored by git.
