"""Repository path anchors.

Import these instead of hardcoding absolute paths, so every script runs from any
working directory and on any machine (portable reproduction + demo for external readers).

    from seal_inspection.paths import ROOT as R          # legacy string form: f"{R}/models/..."
    from seal_inspection.paths import MODELS, DATA        # pathlib.Path form
"""
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
MODELS: Path = REPO_ROOT / "models"
DATA: Path = REPO_ROOT / "data"
OUTPUTS: Path = REPO_ROOT / "outputs"
DOCS: Path = REPO_ROOT / "docs"

# String alias kept for the legacy `R = "/home/ubuntu/TFM/seal-inspection"` usage
# (scripts build paths as f"{R}/models/..." or R + "/...").
ROOT: str = str(REPO_ROOT)
