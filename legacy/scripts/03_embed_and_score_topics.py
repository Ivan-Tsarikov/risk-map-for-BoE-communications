from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the new 03 -> 04 pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--model-size", choices=["small", "base"], default=None)
    parser.add_argument("--run-mode", choices=["dev", "full"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).parent

    build_cmd = [sys.executable, str(script_dir / "03_build_embeddings.py"), "--config", str(args.config)]
    if args.model_size:
        build_cmd.extend(["--model-size", args.model_size])
    if args.run_mode:
        build_cmd.extend(["--run-mode", args.run_mode])

    score_cmd = [sys.executable, str(script_dir / "04_score_and_build_index.py"), "--config", str(args.config)]
    if args.model_size:
        score_cmd.extend(["--model-size", args.model_size])

    print("`03_embed_and_score_topics.py` now runs the new two-step pipeline:")
    print("  1. 03_build_embeddings.py")
    print("  2. 04_score_and_build_index.py")
    subprocess.run(build_cmd, check=True)
    subprocess.run(score_cmd, check=True)


if __name__ == "__main__":
    main()
