from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    replacement = Path(__file__).with_name("04_score_and_build_index.py")
    print("`04_build_index.py` is now a compatibility wrapper for `04_score_and_build_index.py`.")
    runpy.run_path(str(replacement), run_name="__main__")
