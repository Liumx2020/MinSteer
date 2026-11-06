"""Location of this standalone release; no change to experiment text or ordering."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MINSTEER_ROOT", Path(__file__).resolve().parent)).resolve()


def project_path(relative=""):
    return str(ROOT / relative)
