from __future__ import annotations

import argparse
import json
from pathlib import Path

from usig.experiment.analysis import analyze_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pilot uncertainty summaries.")
    parser.add_argument("experiment_id")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_experiment(args.project_root, args.experiment_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
