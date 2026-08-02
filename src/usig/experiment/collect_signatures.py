from __future__ import annotations

import argparse
import json
from pathlib import Path

from usig.experiment.collection import collect, validate_collection

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate or collect Qwen pilot predictions and uncertainty signatures."
    )
    parser.add_argument("action", choices=("validate", "collect", "resume"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "validate":
        result = validate_collection(args.project_root)
        output = {
            "manifest_checksum": result["manifest_checksum"],
            "record_count": len(result["manifest"]),
            "counts": result["counts"],
        }
    else:
        output = collect(args.project_root, limit=args.limit)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
