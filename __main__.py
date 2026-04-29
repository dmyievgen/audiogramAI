"""Bootstrap entry point: ``python audiogram/__main__.py``.

The project uses a ``src/`` layout, so we extend ``sys.path`` once before
importing the real package. This keeps the package importable without
requiring an editable install during development.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from audiogram.app import main  # noqa: E402  — sys.path tweak above

if __name__ == "__main__":
    raise SystemExit(main())
