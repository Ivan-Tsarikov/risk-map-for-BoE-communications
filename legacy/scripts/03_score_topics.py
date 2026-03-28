from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    replacement = Path(__file__).with_name("03_embed_and_score_topics.py")
    print("`03_score_topics.py` is now a compatibility wrapper for `03_embed_and_score_topics.py`.")
    runpy.run_path(str(replacement), run_name="__main__")
