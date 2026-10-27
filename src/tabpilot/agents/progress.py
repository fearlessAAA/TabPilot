"""No-progress tracking of the Execution Agent's tool loop, shared by inference and training rollouts.

A run stops after ``NO_PROGRESS_LIMIT`` consecutive actions that make no progress: a repeated
successful call with identical arguments, or an invalid call that fixes none of the previously
reported field errors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

NO_PROGRESS_LIMIT = 3


def semantic_signature(name: str, arguments: str) -> str:
    """Canonical, argument-order-independent identity for one action call."""
    try:
        normalized = json.dumps(json.loads(arguments), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        normalized = arguments.strip()
    return hashlib.sha256(f"{name}:{normalized}".encode()).hexdigest()


@dataclass
class ProgressTracker:
    """``last_was_repeated_success`` distinguishes a repeated successful call from an unfixed invalid one."""

    prior_obligations: set[tuple[str, str, str | None]] | None = None
    seen_successful_actions: set[str] = field(default_factory=set)
    consecutive_no_progress: int = 0
    last_was_repeated_success: bool = False

    def record(self, tool_name: str, arguments: str, observation_status: str, errors: list[object]) -> bool:
        signature = semantic_signature(tool_name, arguments)
        obligations = {
            (tool_name, str(getattr(error, "code", "")), getattr(error, "path", None)) for error in errors
        }
        progressed = False
        self.last_was_repeated_success = False
        if observation_status == "ok":
            already_seen = signature in self.seen_successful_actions
            progressed = not already_seen
            self.last_was_repeated_success = already_seen
            self.seen_successful_actions.add(signature)
            self.prior_obligations = None
        elif observation_status == "invalid":
            if self.prior_obligations is None:
                progressed = True
            else:
                # Fixing at least one previous field-level obligation is real
                # progress even if a different error remains.
                progressed = bool(self.prior_obligations - obligations)
            self.prior_obligations = obligations
        if progressed:
            self.consecutive_no_progress = 0
        else:
            self.consecutive_no_progress += 1
        return progressed
