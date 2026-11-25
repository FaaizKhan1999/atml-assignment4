#!/usr/bin/env python3
"""Task 1 shared evaluation entry point (see atml4.task1.runner)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atml4.task1.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
