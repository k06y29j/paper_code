#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

CDDM_ROOT = Path(__file__).resolve().parents[2]
if str(CDDM_ROOT) not in sys.path:
    sys.path.insert(0, str(CDDM_ROOT))

from MY.train_cvq.cli import main


if __name__ == "__main__":
    main()

