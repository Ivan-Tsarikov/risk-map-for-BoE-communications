from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    replacement = Path(__file__).with_name("02_filter_candidates.py")
    print("`02_filter_econ.py` is now a compatibility wrapper for `02_filter_candidates.py`.")
    runpy.run_path(str(replacement), run_name="__main__")
