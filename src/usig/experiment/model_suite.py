from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from usig.experiment.repair_v3 import read_jsonl, write_json, write_jsonl

VERSION = "qwen_7b_suite_v1"
MODEL_KEY = "qwen2_5_7b"
MODEL_IDENTIFIER = "Qwen/Qwen2.5-7B-Instruct"


def prepare_calibration_manifest(source: Path, output: Path, count: int = 100) -> dict[str, Any]:
    rows = sorted(
        read_jsonl(source),
        key=lambda item: (item.get("selection_order", 0), item["example_id"]),
    )[:count]
    if len(rows) != count:
        raise ValueError(f"Expected {count} calibration records from {source}")
    rows = [{**row, "selection_order": index} for index, row in enumerate(rows)]
    write_jsonl(output, rows)
    return {
        "version": VERSION,
        "sample_count": len(rows),
        "checksum": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def prepare_all_manifest(normalized: Path, output: Path, dataset: str) -> dict[str, Any]:
    records = sorted(read_jsonl(normalized), key=lambda item: item["example_id"])
    source_checksum = hashlib.sha256(normalized.read_bytes()).hexdigest()
    rows = [
        {
            "canonical_record_checksum": hashlib.sha256(
                (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
            ).hexdigest(),
            "dataset": dataset,
            "example_id": record["example_id"],
            "group_id": record["group_id"],
            "sampling_seed": 2026,
            "sampling_stratum": "full_official_split",
            "selection_order": index,
            "source_checksum": source_checksum,
            "source_split": record["split"],
        }
        for index, record in enumerate(records)
    ]
    write_jsonl(output, rows)
    return {
        "version": VERSION,
        "sample_count": len(rows),
        "checksum": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def projected_size(accuracy: float) -> int | None:
    if not 0 < accuracy < 1:
        return None
    return math.ceil(max(100 / accuracy, 100 / (1 - accuracy)) - 1e-12)


def _correct(item: dict[str, Any], dataset: str) -> bool | None:
    if item.get("unresolved_label"):
        return None
    if dataset == "triviaqa":
        return bool(item["evaluation_diagnostics"]["concise_suffix"]["match"])
    if dataset == "ambignq":
        diagnostics = item.get("evaluation_diagnostics", {})
        return bool(diagnostics.get("any_interpretation_match", False))
    return not bool(item["binary_error"])


def calibration_gate(
    predictions_path: Path,
    verification_path: Path,
    output_path: Path,
    *,
    dataset: str,
    requested_records: int,
    full_available_records: int,
) -> dict[str, Any]:
    predictions = read_jsonl(predictions_path)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    labels = [_correct(item, dataset) for item in predictions]
    resolved = [item for item in labels if item is not None]
    counts = Counter(resolved)
    truncation_count = sum(item.get("token_limit_reached", False) for item in predictions)
    parsing_failure_count = sum(
        item.get("evaluation_metrics", {}).get("parsing_status") not in (None, "ok")
        for item in predictions
    )
    accuracy = counts[True] / len(resolved) if resolved else 0.0
    projected = projected_size(accuracy)
    artifact_failures = sum(
        value.get("checksum_failure_count", 0)
        + value.get("missing_count", 0)
        + value.get("unexpected_count", 0)
        + value.get("non_finite_feature_count", 0)
        for value in verification["artifacts"].values()
    )
    reasons = []
    if len(predictions) != requested_records:
        reasons.append("record_count_mismatch")
    if truncation_count / max(1, len(predictions)) > 0.05:
        reasons.append("truncation_above_5_percent")
    if parsing_failure_count / max(1, len(predictions)) > 0.05:
        reasons.append("parsing_failure_above_5_percent")
    if artifact_failures or not verification.get("complete"):
        reasons.append("artifact_verification_failure")
    if projected is None:
        reasons.append("one_class_calibration")
    elif projected > full_available_records:
        reasons.append("projected_minority_below_100")
    result = {
        "version": VERSION,
        "model_key": MODEL_KEY,
        "model_identifier": MODEL_IDENTIFIER,
        "dataset": dataset,
        "requested_records": requested_records,
        "collected_records": len(predictions),
        "valid_predictions": verification["artifacts"]["predictions"]["valid_count"],
        "valid_ifi_signatures": min(
            verification["artifacts"]["compact_signatures"]["valid_count"],
            verification["artifacts"]["signature_ablations"]["valid_count"],
        ),
        "truncation_count": truncation_count,
        "truncation_rate": truncation_count / max(1, len(predictions)),
        "parsing_failure_count": parsing_failure_count,
        "parsing_failure_rate": parsing_failure_count / max(1, len(predictions)),
        "correct_count": counts[True],
        "incorrect_count": counts[False],
        "unresolved_count": sum(item is None for item in labels),
        "minority_class_count": min(counts[True], counts[False]),
        "accuracy": accuracy,
        "projected_full_collection_size": projected,
        "full_available_records": full_available_records,
        "projected_minority_class_at_available_size": min(
            accuracy, 1 - accuracy
        ) * full_available_records,
        "artifact_failure_count": artifact_failures,
        "gate_passed": not reasons,
        "gate_reasons": reasons or ["passed"],
    }
    write_json(output_path, result)
    return result


def require_gate(path: Path) -> dict[str, Any]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("version") != VERSION or not gate.get("gate_passed"):
        raise PermissionError(
            f"Model dataset run refused: gate failed ({', '.join(gate.get('gate_reasons', []))})"
        )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    manifest = sub.add_parser("calibration-manifest")
    manifest.add_argument("--source", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--count", type=int, default=100)
    full = sub.add_parser("full-manifest")
    full.add_argument("--normalized", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True)
    full.add_argument("--dataset", required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--predictions", type=Path, required=True)
    gate.add_argument("--verification", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--dataset", required=True)
    gate.add_argument("--requested-records", type=int, required=True)
    gate.add_argument("--full-available-records", type=int, required=True)
    require = sub.add_parser("require-gate")
    require.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "calibration-manifest":
        result = prepare_calibration_manifest(args.source, args.output, args.count)
    elif args.action == "full-manifest":
        result = prepare_all_manifest(args.normalized, args.output, args.dataset)
    elif args.action == "gate":
        result = calibration_gate(
            args.predictions,
            args.verification,
            args.output,
            dataset=args.dataset,
            requested_records=args.requested_records,
            full_available_records=args.full_available_records,
        )
    else:
        result = require_gate(args.gate)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
