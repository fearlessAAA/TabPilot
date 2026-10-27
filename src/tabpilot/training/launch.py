"""Run ``swift rlhf`` after installing an import alias that TRL needs with newer vLLM releases.

Usage: ``python -m tabpilot.training.launch <swift rlhf arguments>``.
"""

from __future__ import annotations

import runpy
import sys
import types


def install_trl_vllm_compat() -> None:
    """TRL imports ``vllm.transformers_utils.tokenizer``, which newer vLLM releases moved to
    ``vllm.tokenizers.hf``; alias it in memory when it is missing."""
    try:
        import vllm.transformers_utils.tokenizer  # noqa: F401
    except ImportError:
        import vllm.transformers_utils as transformers_utils
        from vllm.tokenizers import hf

        module = types.ModuleType("vllm.transformers_utils.tokenizer")
        module.get_cached_tokenizer = hf.get_cached_tokenizer
        sys.modules[module.__name__] = module
        transformers_utils.tokenizer = module
    import importlib

    # Optional TRL integrations that are never used here and break on partial installs.
    try:
        import_utils = importlib.import_module("trl.import_utils")
    except ImportError:
        return
    for name in ("_weave_available", "_mergekit_available", "_llm_blender_available"):
        if hasattr(import_utils, name):
            setattr(import_utils, name, False)


if __name__ == "__main__":
    install_trl_vllm_compat()
    sys.argv[0] = "swift rlhf"
    runpy.run_module("swift.cli.rlhf", run_name="__main__")
