"""Persist experiment/evaluation results as JSON under outputs/results/.

Every experiment calls save_results(name, data) so its numbers are machine-readable
and auditable (mirrors what REPRODUCE.md reports). Handles numpy scalars.
"""
import json
import numpy as np
from .paths import REPO_ROOT


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def save_results(name: str, data: dict):
    """Write `data` to outputs/results/<name>.json and echo the path."""
    d = REPO_ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default)
    print(f"[results] saved {p}", flush=True)
    return p
