"""In-memory compatibility fixes for the TFM libraries; model code on disk is never edited.

TabPFN 9.0.0 (v3.5 embedding initialization):
``TrainableOrthogonalEmbedding._init`` runs ``torch.linalg.qr`` on a CPU tensor, which fails on
PyTorch builds without CPU LAPACK (e.g. some ROCm builds). The patched method builds the QR input
on the model's device; model state is otherwise unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def install_tabpfn_v3_5_qr_device_fix(*, device: str = "cuda") -> None:
    """Make ``TrainableOrthogonalEmbedding._init`` build its QR input on ``device``.

    A no-op if the module is not importable (TabPFN not installed) or the
    patch was already applied. CPU-only environments are unaffected in
    practice: only calling this before constructing a V3.5 model on a
    LAPACK-less CPU build changes anything, and the patched method still
    ends by copying results into the module's existing (CPU) parameter
    tensors, so model state is unchanged either way.
    """
    module_name = "tabpfn.architectures.tabpfn_v3_5"
    if module_name in sys.modules and getattr(
        sys.modules[module_name].TrainableOrthogonalEmbedding, "_tabpilot_qr_device_patched", False
    ):
        return

    import torch
    from tabpfn.architectures import tabpfn_v3_5

    embedding_cls = tabpfn_v3_5.TrainableOrthogonalEmbedding

    def _init(self: "tabpfn_v3_5.TrainableOrthogonalEmbedding") -> None:
        weight = self.embedding.weight
        num_classes, embed_dim = weight.shape
        k = min(num_classes, embed_dim)
        q, _ = torch.linalg.qr(torch.randn(embed_dim, k, device=device))
        ortho_rows = q.T.to(weight.device)  # (k, embed_dim)
        with torch.no_grad():
            weight[:k].copy_(ortho_rows)
            if num_classes > embed_dim:
                extra = torch.randn(num_classes - k, embed_dim, device=device)
                extra = extra / extra.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                weight[k:].copy_(extra.to(weight.device))

    embedding_cls._init = _init  # type: ignore[method-assign]
    embedding_cls._tabpilot_qr_device_patched = True


def install_limix_cache_dir_fix() -> None:
    """Redirect LimiX 2.0.0's hardcoded preprocessing cache directory in memory.

    ``LimiXPredictor.__init__`` always builds its ``CacheManager`` with a developer-machine path
    (``/mnt/public/...``) and ``CacheManager.__init__`` runs ``os.makedirs`` on it even when the
    cache is disabled, which fails wherever that path is not writable. Only the directory changes;
    preprocessing, inference and the (disabled) cache semantics are unchanged.
    """
    from inference.v2_0 import predictor as limix_v2

    from tabpilot.paths import DEFAULT_TFM_WEIGHTS_ROOT

    manager = limix_v2.LimiXPredictor.CacheManager
    if getattr(manager, "_tabpilot_cache_dir_patched", False):
        return
    original_init = manager.__init__
    redirected = Path(os.environ.get("TABPILOT_TFM_WEIGHTS", DEFAULT_TFM_WEIGHTS_ROOT)) / "limix" / "preprocess_cache"

    def patched_init(self, cache_dir: str | None = None) -> None:
        original_init(self, cache_dir=str(redirected))

    manager.__init__ = patched_init
    manager._tabpilot_cache_dir_patched = True
