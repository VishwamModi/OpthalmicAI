from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from backend.main import app  # re-export for platform entrypoints

__all__ = ["app"]
