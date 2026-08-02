from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from usig.experiment.repair_v3 import write_json


def build(model_root: Path, output_directory: Path) -> dict[str, Any]:
    gates = {}
    for path in sorted((model_root / "calibration").glob("*/calibration_gate.json")):
        gates[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    datasets = {}
    for destination in sorted((model_root / "full").glob("*")):
        if not destination.is_dir():
            continue
        metadata = destination / "extraction_metadata/experiment.json"
        checksums = destination / "verification_reports/artifact_checksums.json"
        analyses = sorted(destination.glob("analysis/*.json"))
        datasets[destination.name] = {
            "metadata": json.loads(metadata.read_text()) if metadata.exists() else None,
            "checksums": json.loads(checksums.read_text()) if checksums.exists() else None,
            "analyses": {
                path.name: json.loads(path.read_text(encoding="utf-8"))
                for path in analyses
            },
        }
    for destination in sorted((model_root / "repairs").glob("*/full")):
        if not destination.is_dir():
            continue
        key = f"repair_{destination.parent.name}"
        metadata = destination / "extraction_metadata/experiment.json"
        checksums = destination / "verification_reports/artifact_checksums.json"
        analyses = sorted(destination.glob("analysis/*.json"))
        datasets[key] = {
            "metadata": json.loads(metadata.read_text()) if metadata.exists() else None,
            "checksums": json.loads(checksums.read_text()) if checksums.exists() else None,
            "analyses": {
                path.name: json.loads(path.read_text(encoding="utf-8"))
                for path in analyses
            },
        }
    result = {
        "version": "qwen_7b_report_v1",
        "model_root": str(model_root),
        "calibration_gates": gates,
        "datasets": datasets,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    index = len(list(output_directory.glob("report_*.json"))) + 1
    json_path = output_directory / f"report_{index:03d}.json"
    write_json(json_path, result)
    markdown_path = output_directory / f"report_{index:03d}.md"
    if markdown_path.exists():
        raise FileExistsError(markdown_path)
    markdown_path.write_text(
        "# Qwen2.5-7B IFI report\n\n"
        f"Calibration gates: {len(gates)}\n\n"
        f"Full dataset directories: {len(datasets)}\n\n"
        f"JSON checksum: `{hashlib.sha256(json_path.read_bytes()).hexdigest()}`\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(markdown_path), **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.model_root, args.output_directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
